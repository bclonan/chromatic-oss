"""Verify native RGBD hue algebra against an independent integer NumPy oracle."""
from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULUS = 391680
ROTATION = 123457


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def phase_digest(phases):
    """Hash phase integers in a fixed little-endian uint32 representation."""
    values = array("I", phases)
    if values.itemsize != 4:
        raise RuntimeError("Phase hashing requires a four-byte unsigned integer")
    if sys.byteorder != "little":
        values.byteswap()
    return digest_bytes(values.tobytes())


def reference_rgbd(phases):
    """Encode the specified 1,530-color wheel, independent of the native binding."""
    import numpy as np
    values = np.asarray(phases, dtype=np.uint32).reshape(-1)
    hue, fine = values // 256, values % 256
    sector, offset = hue // 255, hue % 255
    output = np.empty((len(values), 4), dtype=np.uint8)
    corners = ((255, 0, 0), (255, 255, 0), (0, 255, 0),
               (0, 255, 255), (0, 0, 255), (255, 0, 255))
    for index, first in enumerate(corners):
        mask = sector == index
        last = corners[(index + 1) % 6]
        for channel in range(3):
            if first[channel] == last[channel]:
                output[mask, channel] = first[channel]
            elif first[channel] == 0:
                output[mask, channel] = offset[mask]
            else:
                output[mask, channel] = 255 - offset[mask]
    output[:, 3] = fine
    return output.tobytes()


def reference_blend(a, b, weight):
    import numpy as np
    first, second = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    forward = (second - first) % MODULUS
    delta = np.where(forward <= MODULUS // 2, forward, forward - MODULUS)
    step = np.sign(delta) * ((np.abs(delta) * weight + 127) // 255)
    return ((first + step) % MODULUS).astype(np.uint32)


def reference_operations(a, b):
    import numpy as np
    return {
        "add": lambda: ((a.astype(np.uint64) + b.astype(np.uint64)) % MODULUS).astype(np.uint32),
        "subtract": lambda: ((a.astype(np.int64) - b.astype(np.int64)) % MODULUS).astype(np.uint32),
        "multiply": lambda: ((a.astype(np.uint64) * b.astype(np.uint64)) % MODULUS).astype(np.uint32),
        "rotate": lambda: ((a.astype(np.uint64) + ROTATION) % MODULUS).astype(np.uint32),
        "blend": lambda: reference_blend(a, b, 127),
        "sum": lambda: np.asarray(int(a.sum(dtype=np.uint64)) % MODULUS, dtype=np.uint32),
        "dot": lambda: np.asarray(int((a.astype(np.uint64) * b.astype(np.uint64)).sum(dtype=np.uint64)) % MODULUS, dtype=np.uint32),
    }


def native_operations(a, b):
    return {"add": lambda: a + b, "subtract": lambda: a - b, "multiply": lambda: a * b,
            "rotate": lambda: a.rotate(ROTATION), "blend": lambda: a.blend(b, 127),
            "sum": lambda: a.sum(), "dot": lambda: a.dot(b)}


def assert_native_exact(tensor, expected):
    import numpy as np
    expected = np.asarray(expected, dtype=np.uint32)
    if (tensor.shape != expected.shape or tensor.storage_nbytes != expected.size * 4
            or phase_digest(tensor.phases()) != digest_bytes(expected.astype("<u4").tobytes())
            or tensor.rgbd_bytes != reference_rgbd(expected)):
        raise ValueError("Native RGBD bytes or phase algebra disagree with the integer oracle")


def measured_call(function, repeats, native=False):
    result = None
    started = time.perf_counter_ns()
    for _ in range(repeats):
        if native and result is not None:
            result.close()
        result = function()
    return result, time.perf_counter_ns() - started


def file_only_cold_load(path, expected_rgbd, expected_phases, expected_rotation, expected_binary):
    """This branch imports neither NumPy nor PyTorch and receives no source colors."""
    from chromatensor import ColorTensor, HueTensor
    from chromatensor.hue import native_info
    info = native_info()
    if info["sha256"] != expected_binary or ColorTensor is not HueTensor:
        raise ValueError("Cold process native binary or ColorTensor alias differs")
    with HueTensor.load(path) as loaded:
        rgbd_hash, phase_hash = digest_bytes(loaded.rgbd_bytes), phase_digest(loaded.phases())
        with loaded.rotate(ROTATION) as rotated:
            rotated_hash = phase_digest(rotated.phases())
        if (rgbd_hash, phase_hash, rotated_hash) != (expected_rgbd, expected_phases, expected_rotation):
            raise ValueError("File-only cold load or subsequent native rotation differs")
        result = {"shape": list(loaded.shape), "examples": loaded.numel, "payload_bytes": loaded.storage_nbytes,
                  "rgbd_sha256": rgbd_hash, "phase_sha256": phase_hash, "rotated_phase_sha256": rotated_hash}
    if any(name == "torch" or name.startswith("torch.") or name == "numpy" or name.startswith("numpy.")
           for name in sys.modules):
        raise RuntimeError("Native-only cold process imported a tensor dependency")
    info["path"] = relative(info["path"])
    return {**result, "native_binary": info, "torch_imported": False, "numpy_imported": False,
            "input_scope": "saved HueTensor file and reference hashes only", "isolated_python": sys.flags.isolated == 1}


def cold_process(path, rgbd_hash, phase_hash, rotation_hash, binary_hash):
    command = [sys.executable, "-I", relative(Path(__file__)), "--cold-load", relative(path),
               "--expected-rgbd-sha256", rgbd_hash, "--expected-phase-sha256", phase_hash,
               "--expected-rotation-sha256", rotation_hash, "--expected-binary-sha256", binary_hash]
    started = time.perf_counter_ns()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", timeout=120, check=True)
    result = json.loads(process.stdout)
    if not result["isolated_python"] or result["torch_imported"] or result["numpy_imported"]:
        raise ValueError("Cold process did not use the isolated native-only path")
    from scripts.build_native_hue import public_text
    return result, {"arguments": command[2:], "elapsed_ns": time.perf_counter_ns() - started,
                    "returncode": process.returncode, "stderr": public_text(process.stderr)}


def relative(path, root=ROOT):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def resolve(path, root=ROOT):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (Path(root) / value).resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_binary_info(info, root=ROOT):
    path = resolve(info["path"], root)
    if (info.get("abi_version") != 1 or info.get("phase_modulus") != MODULUS
            or info.get("runtime") != "Rust integer color-token operators"
            or path.stat().st_size != info["bytes"] or digest_bytes(path.read_bytes()) != info["sha256"]):
        raise ValueError("Native binary identity or ABI differs")


def matrix_and_mutation_checks(a_values, b_values, HueTensor):
    import numpy as np
    width = min(32, math.isqrt(len(a_values)))
    first = a_values[:width * width].reshape(width, width)
    second = b_values[:width * width].reshape(width, width)
    expected = ((first.astype(np.uint64) @ second.astype(np.uint64)) % MODULUS).astype(np.uint32)
    with HueTensor.from_phases(first) as a, HueTensor.from_phases(second) as b:
        with a @ b as product:
            assert_native_exact(product, expected)
            original_product = product.rgbd_bytes
        with a.permute(1, 0) as transposed:
            assert_native_exact(transposed, first.T)
        with a.slice(1, 0, width, 2) as sliced:
            assert_native_exact(sliced, first[:, ::2])
        with a.reshape(width * width) as flattened:
            assert_native_exact(flattened, first.reshape(-1))
        mutated_bytes = bytearray(a.rgbd_bytes)
        mutated_bytes[3] ^= 1
        mutated = first.copy()
        mutated.flat[0] ^= np.uint32(1)
        with HueTensor.from_rgbd(mutated_bytes, a.shape) as changed:
            assert_native_exact(changed, mutated)
            with changed @ b as changed_product:
                changed_expected = ((mutated.astype(np.uint64) @ second.astype(np.uint64)) % MODULUS).astype(np.uint32)
                assert_native_exact(changed_product, changed_expected)
                if changed_product.rgbd_bytes == original_product:
                    raise ValueError("Changing a D byte did not change the measured matrix contraction")
        changed_rgb = first.copy()
        changed_rgb.flat[0] = (int(changed_rgb.flat[0]) + 256) % MODULUS
        with HueTensor.from_rgbd(reference_rgbd(changed_rgb), a.shape) as changed:
            assert_native_exact(changed, changed_rgb)
    return {"shape": [width, width], "matmul_phase_sha256": digest_bytes(expected.astype("<u4").tobytes()),
            "D_byte_mutation_changes_matmul": True, "RGB_byte_mutation_changes_phase": True,
            "reshape_permute_slice_exact": True}


def synthetic_retrieval(HueTensor, seed):
    """A synthetic dictionary task, with no learned parameters or neural claims."""
    import numpy as np
    generator = np.random.default_rng(seed + 1)
    templates = generator.integers(0, MODULUS, size=(8, 32), dtype=np.uint32)
    templates[:, 0] = 0
    labels = generator.integers(0, 8, size=256, dtype=np.uint32)
    shifts = generator.integers(0, MODULUS, size=256, dtype=np.uint32)
    dictionary = {tuple(int(value) for value in row): label for label, row in enumerate(templates)}
    started = time.perf_counter_ns()
    native_predictions = []
    for label, shift in zip(labels, shifts):
        with HueTensor.from_phases(templates[int(label)]) as template:
            with template.rotate(int(shift)) as query:
                with HueTensor.from_phases(query.phases()[0]) as origin:
                    with query - origin as signature:
                        native_predictions.append(dictionary[signature.phases()])
    native_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    queries = ((templates[labels].astype(np.uint64) + shifts[:, None]) % MODULUS).astype(np.uint32)
    signatures = ((queries.astype(np.int64) - queries[:, :1].astype(np.int64)) % MODULUS).astype(np.uint32)
    control_predictions = [dictionary[tuple(int(value) for value in row)] for row in signatures]
    control_ns = time.perf_counter_ns() - started
    if native_predictions != control_predictions or native_predictions != labels.tolist():
        raise ValueError("Synthetic phase-invariant dictionary retrieval differs")
    return {"source": "deterministic synthetic eight-template color phase dictionary", "seed": seed + 1,
            "examples": len(labels), "template_shape": [8, 32], "correct": len(labels),
            "native_elapsed_ns": native_ns, "uint32_control_elapsed_ns": control_ns,
            "logical_template_payload_bytes_per_representation": int(templates.size * 4),
            "timing_scope": "native per-query construction, rotation, subtraction, copy and dictionary lookup; NumPy uses a vectorized batch and the same lookup",
            "scope": "native rotation and subtraction; Python exact dictionary lookup; no training or neural inference"}


def run_verification(output, artifacts=None, *, samples=1000000, seed=7331, repeats=3):
    import numpy as np
    from chromatensor import ColorTensor, HueTensor
    from chromatensor.hue import native_info
    from experiments.provenance import artifact_manifest, capture_provenance, verify_provenance
    if type(samples) is not int or not 16 <= samples <= 64 * 1024**2 or type(seed) is not int or not 0 <= seed < 2**64 or type(repeats) is not int or repeats < 1:
        raise ValueError("Use 16 to 64M samples, an unsigned integer seed and positive repeats")
    if ColorTensor is not HueTensor:
        raise ValueError("ColorTensor must be the native HueTensor alias")
    run_id = "native-hue-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12]
    directory = Path(artifacts) if artifacts is not None else ROOT / "artifacts/native-hue" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    config = {"profile": "native_hue" if samples >= 1000000 else "native_hue_smoke", "samples": samples,
              "seed": seed, "repeats": repeats, "phase_modulus": MODULUS, "payload_bits_per_value": 32}
    binary = native_info()
    verify_binary_info(binary)
    provenance = capture_provenance(config, artifact_paths=[binary["path"]])
    generator = np.random.default_rng(seed)
    a_values = generator.integers(0, MODULUS, size=samples, dtype=np.uint32)
    b_values = generator.integers(0, MODULUS, size=samples, dtype=np.uint32)
    first_path, second_path = directory / "reference-phases.u32", directory / "second-phases.u32"
    first_path.write_bytes(a_values.astype("<u4").tobytes())
    second_path.write_bytes(b_values.astype("<u4").tobytes())
    control_metadata = directory / "uint32-control.json"
    write_json(control_metadata, {"shape": [samples], "dtype": "uint32 little-endian", "phase_modulus": MODULUS})
    operations = {}
    references = reference_operations(a_values, b_values)
    with HueTensor.from_phases(a_values) as a, HueTensor.from_phases(b_values) as b:
        assert_native_exact(a, a_values)
        assert_native_exact(b, b_values)
        functions = native_operations(a, b)
        for name, function in functions.items():
            measured, native_ns = measured_call(function, repeats, native=True)
            try:
                expected, control_ns = measured_call(references[name], repeats)
                assert_native_exact(measured, expected)
                operations[name] = {"shape": list(measured.shape), "examples": measured.numel,
                                    "phase_sha256": phase_digest(measured.phases()),
                                    "rgbd_sha256": digest_bytes(measured.rgbd_bytes), "exact": True,
                                    "native_elapsed_ns": native_ns, "uint32_control_elapsed_ns": control_ns,
                                    "repeats": repeats}
            finally:
                measured.close()
        tensor_path = directory / "million-colors.huetensor"
        file_bytes = a.save(tensor_path)
        rgbd_hash, phase_hash = digest_bytes(a.rgbd_bytes), phase_digest(a.phases())
    matrix = matrix_and_mutation_checks(a_values, b_values, HueTensor)
    retrieval = synthetic_retrieval(HueTensor, seed)
    cold, process = cold_process(tensor_path, rgbd_hash, phase_hash,
                                 digest_bytes(references["rotate"]().astype("<u4").tobytes()), binary["sha256"])
    cold_path = directory / "cold-process.json"
    write_json(cold_path, cold)
    source_files = list((ROOT / "native/hue_tensor/src").rglob("*.rs"))
    if not source_files or any(re.search(r"\bf(?:32|64)\b", path.read_text(encoding="utf-8")) for path in source_files):
        raise ValueError("Native core source contains floating point types or is absent")
    if native_info() != binary:
        raise ValueError("Native binary changed during verification")
    verify_provenance(provenance, config)
    paths = [path for path in directory.iterdir() if path.is_file()] + [Path(binary["path"])]
    provenance["artifacts"] = artifact_manifest(paths)
    result = {"schema_version": 1, "run_id": run_id, "status": "complete",
              "created_at": provenance["captured_at"], "completed_at": datetime.now(timezone.utc).isoformat(),
              "config": config, "provenance": provenance, "native_binary": {**binary, "path": relative(binary["path"])},
              "execution": {"runtime": "Rust integer color-token operators", "tensor_payload": "four uint8 RGBD bytes",
                            "phase_domain": "integers modulo 391680", "floating_tensor_core": False,
                            "native_core_source_floating_types_absent": True, "autograd": False,
                            "oracle": "NumPy uint32 payloads with uint64/int64 arithmetic temporaries",
                            "timing_scope": "operator calls and output allocation; integer control includes dtype promotion; input construction, validation and persistence excluded"},
              "tensor_file": {"path": relative(tensor_path), "sha256": digest_bytes(tensor_path.read_bytes()),
                              "bytes": file_bytes, "payload_bytes": samples * 4,
                              "container_overhead_bytes": file_bytes - samples * 4, "rgbd_sha256": rgbd_hash,
                              "phase_sha256": phase_hash, "shape": [samples]},
              "integer_control": {"path": relative(first_path), "second_path": relative(second_path),
                                  "metadata_path": relative(control_metadata), "payload_bytes": first_path.stat().st_size,
                                  "metadata_bytes": control_metadata.stat().st_size,
                                  "total_bytes": first_path.stat().st_size + control_metadata.stat().st_size,
                                  "sha256": digest_bytes(first_path.read_bytes())},
              "operations": operations, "matrix_and_mutations": matrix, "synthetic_retrieval": retrieval,
              "cold_process": {"path": relative(cold_path), "receipt": cold, **process},
              "non_claims": ["This is finite hue ring arithmetic, not real-valued tensor arithmetic.",
                             "The color bytes relabel the same modular phases as the 32-bit integer control.",
                             "Synthetic dictionary retrieval does not establish learning or real-data generalization.",
                             "Measured CPU operator timings do not establish optimized hardware or energy gains."]}
    validate_native_hue_receipt(result, require_million=samples >= 1000000)
    write_json(Path(output).parent / "runs" / (run_id + ".json"), result)
    write_json(output, result)
    return result


def validate_native_hue_receipt(result, root=ROOT, *, require_million=True):
    import numpy as np
    from chromatensor import HueTensor
    from chromatensor.hue import native_info
    from experiments.provenance import verify_provenance
    config = result["config"]
    if (result.get("schema_version") != 1 or result.get("status") != "complete"
            or type(config.get("samples")) is not int or not 16 <= config["samples"] <= 64 * 1024**2
            or type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**64
            or type(config.get("repeats")) is not int or config["repeats"] < 1
            or config.get("phase_modulus") != MODULUS or config.get("payload_bits_per_value") != 32
            or require_million and (config.get("samples", 0) < 1000000 or config.get("seed") != 7331
                                    or config.get("profile") != "native_hue")):
        raise ValueError("Native hue evidence is incomplete or reduced")
    execution = result.get("execution", {})
    if (execution.get("runtime") != "Rust integer color-token operators"
            or execution.get("tensor_payload") != "four uint8 RGBD bytes"
            or execution.get("phase_domain") != "integers modulo 391680"
            or execution.get("floating_tensor_core") is not False or execution.get("autograd") is not False
            or execution.get("native_core_source_floating_types_absent") is not True):
        raise ValueError("Native hue execution claims differ from the verified contract")
    source_files = list((Path(root) / "native/hue_tensor/src").rglob("*.rs"))
    if not source_files or any(re.search(r"\bf(?:32|64)\b", path.read_text(encoding="utf-8")) for path in source_files):
        raise ValueError("Native core source contains floating point types or is absent")
    binary = result["native_binary"]
    verify_binary_info(binary, root)
    if binary["sha256"] != native_info()["sha256"]:
        raise ValueError("Native binary identity differs from the loaded library")
    required = [binary["path"], result["tensor_file"]["path"], result["integer_control"]["path"],
                result["integer_control"]["second_path"], result["integer_control"]["metadata_path"],
                result["cold_process"]["path"]]
    verify_provenance(result["provenance"], config, root, required_artifacts=required)
    generator = np.random.default_rng(config["seed"])
    a_values = generator.integers(0, MODULUS, size=config["samples"], dtype=np.uint32)
    b_values = generator.integers(0, MODULUS, size=config["samples"], dtype=np.uint32)
    control = result["integer_control"]
    metadata_path = resolve(control["metadata_path"], root)
    if (resolve(control["path"], root).read_bytes() != a_values.astype("<u4").tobytes()
            or resolve(control["second_path"], root).read_bytes() != b_values.astype("<u4").tobytes()
            or control["payload_bytes"] != config["samples"] * 4
            or control["sha256"] != digest_bytes(a_values.astype("<u4").tobytes())
            or control["metadata_bytes"] != metadata_path.stat().st_size
            or json.loads(metadata_path.read_text(encoding="utf-8")) !=
                {"shape": [config["samples"]], "dtype": "uint32 little-endian", "phase_modulus": MODULUS}
            or control["total_bytes"] != control["payload_bytes"] + control["metadata_bytes"]):
        raise ValueError("Integer control does not match the seeded phase source or physical storage")
    references = reference_operations(a_values, b_values)
    file = result["tensor_file"]
    path = resolve(file["path"], root)
    if (path.stat().st_size != file["bytes"] or digest_bytes(path.read_bytes()) != file["sha256"]
            or file["payload_bytes"] != config["samples"] * 4
            or file["container_overhead_bytes"] != file["bytes"] - file["payload_bytes"]):
        raise ValueError("Native color file physical storage changed")
    with HueTensor.load(path) as a, HueTensor.from_phases(b_values) as b:
        assert_native_exact(a, a_values)
        if set(result["operations"]) != set(references):
            raise ValueError("Required native algebra operation evidence is missing")
        for name, function in native_operations(a, b).items():
            expected = references[name]()
            with function() as actual:
                assert_native_exact(actual, expected)
                row = result["operations"][name]
                if (row["shape"] != list(actual.shape) or row["examples"] != actual.numel
                        or row["phase_sha256"] != phase_digest(actual.phases())
                        or row["rgbd_sha256"] != digest_bytes(actual.rgbd_bytes) or row["exact"] is not True
                        or any(type(row[key]) is not int or row[key] <= 0 for key in
                               ("native_elapsed_ns", "uint32_control_elapsed_ns", "repeats"))):
                    raise ValueError("Recorded native algebra operation disagrees with replay")
        if file["rgbd_sha256"] != digest_bytes(a.rgbd_bytes) or file["phase_sha256"] != phase_digest(a.phases()):
            raise ValueError("Native saved-color hashes disagree with replay")
    if matrix_and_mutation_checks(a_values, b_values, HueTensor) != result["matrix_and_mutations"]:
        raise ValueError("Native matrix and actual color mutation evidence differs")
    actual_retrieval = synthetic_retrieval(HueTensor, config["seed"])
    measured_retrieval = result["synthetic_retrieval"]
    timing_fields = {"native_elapsed_ns", "uint32_control_elapsed_ns"}
    if ({key: value for key, value in actual_retrieval.items() if key not in timing_fields} !=
            {key: value for key, value in measured_retrieval.items() if key not in timing_fields}
            or any(type(measured_retrieval[key]) is not int or measured_retrieval[key] <= 0 for key in timing_fields)):
        raise ValueError("Native synthetic retrieval evidence differs from independent replay")
    cold_file = json.loads(resolve(result["cold_process"]["path"], root).read_text(encoding="utf-8"))
    if cold_file != result["cold_process"]["receipt"]:
        raise ValueError("Cold-process receipt artifact differs")
    replay, _ = cold_process(path, file["rgbd_sha256"], file["phase_sha256"],
                             digest_bytes(references["rotate"]().astype("<u4").tobytes()), binary["sha256"])
    if replay != cold_file:
        raise ValueError("File-only native cold-process replay differs")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/native-hue-verification.json")
    parser.add_argument("--samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cold-load", type=Path)
    for name in ("rgbd", "phase", "rotation", "binary"):
        parser.add_argument("--expected-" + name + "-sha256")
    args = parser.parse_args(argv)
    if args.cold_load:
        hashes = [getattr(args, "expected_" + name + "_sha256") for name in ("rgbd", "phase", "rotation", "binary")]
        if any(value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            parser.error("Cold loading requires four reference SHA256 hashes")
        print(json.dumps(file_only_cold_load(args.cold_load, *hashes)), flush=True)
    else:
        result = run_verification(args.output, samples=args.samples, seed=args.seed, repeats=args.repeats)
        print(json.dumps({"run_id": result["run_id"], "samples": result["config"]["samples"],
                          "native_binary": result["native_binary"]["sha256"], "cold_load": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
