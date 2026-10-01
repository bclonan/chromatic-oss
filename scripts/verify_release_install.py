"""Check an installed wheel under isolated Python, including real saved inference."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import platform
import site
import sys
import tempfile
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    import chromatensor
    from chromatensor import ChromaCube, HolographicCube, UniformCodec

    torch.set_num_threads(4)
    prefix = Path(sys.prefix).resolve()
    modules = {}
    assets = {}
    for name in ("chromatensor", "experiments", "dashboard"):
        module = __import__(name)
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(prefix):
            raise RuntimeError("Package import reused an external installation: " + str(location))
        modules[name] = "<environment>/" + location.relative_to(prefix).as_posix()
        for path in location.parent.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".js", ".css", ".html"}:
                assets[name + "/" + path.relative_to(location.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if site.ENABLE_USER_SITE or sys.flags.isolated != 1:
        raise RuntimeError("Release imports must run with isolated Python and user site disabled")
    for name in ("torch", "numpy", "torchvision", "matplotlib", "psutil"):
        imported = __import__(name)
        if not Path(imported.__file__).resolve().is_relative_to(prefix):
            raise RuntimeError("Dependency import reused host packages: " + name)

    values = torch.linspace(-1, 1, 27).reshape(3, 3, 3)
    cube = ChromaCube.from_tensor(values, UniformCodec(16))
    restored = cube.twist().inverse_twist()
    inverse = torch.equal(restored.tensor.codes.view(torch.uint8), cube.tensor.codes.view(torch.uint8))
    hologram = HolographicCube.from_tensor(values)
    finite = bool(torch.isfinite(hologram.decode()).all())
    required = ("index.html", "app.js", "style.css", "cube.js", "cube.css")
    bundled = all(resources.files("dashboard").joinpath(name).is_file() for name in required)
    if not inverse or not finite or not bundled or metadata.version("chromatensor") != "0.2.0":
        raise RuntimeError("Installed release API or asset verification failed")

    from chromatensor import ChromaTokenizer, ChromaDecoder, TokenSequence
    from chromatensor.autograd import ChromaQuantizer
    values = torch.tensor([1., -1., 1.5, 2.])
    sequence = ChromaTokenizer().encode(values)
    expected_colors = [[0, 0, 0, 127], [128, 0, 0, 127], [64, 0, 0, 127], [0, 0, 0, 128]]
    decoder = ChromaDecoder()
    with tempfile.TemporaryDirectory() as directory:
        sequence.save(Path(directory) / "tokens.ctensor", compress=False)
        persisted = TokenSequence.load(Path(directory) / "tokens.ctensor")
    decoded = decoder.decode(persisted)
    changed = TokenSequence(sequence.tokens[:1].clone(), (1,), sequence.codec_config)
    changed.tokens[0, 3] = 128
    mutated_value = float(decoder.decode(changed)[0])
    quantizer = ChromaQuantizer(representation="rgbd")
    neural = quantizer(values)
    rgbd_ok = (sequence.tokens.dtype == torch.uint8 and sequence.tokens.tolist() == expected_colors
               and persisted.tokens.tolist() == expected_colors and torch.equal(decoded, values)
               and mutated_value == 2. and torch.equal(neural, values)
               and quantizer.last_state.rgbd.tolist() == expected_colors)
    if not rgbd_ok:
        raise RuntimeError("Installed RGBD encoding, byte mutation, persistence or neural decode failed")
    rgbd_encoding = {"tokens": sequence.tokens.tolist(), "decoded": decoded.tolist(),
                     "dtype": str(sequence.tokens.dtype), "payload_bytes": sequence.storage_nbytes,
                     "mutated_D_decode": mutated_value, "persisted_tokens": persisted.tokens.tolist(),
                     "neural_tokens": quantizer.last_state.rgbd.tolist(),
                     "neural_range_metadata_bytes": quantizer.last_state.range_metadata_bytes}

    inference = []
    if args.results:
        from experiments.checkpoint_probe import build_checkpoint_model, load_numeric_npz
        from experiments.common import autocast_context
        from experiments.models import convert_integer_cpu
        result = json.loads(args.results.read_text(encoding="utf-8"))
        root = args.results.resolve().parents[1]
        for representation in ("fp32", "chroma8"):
            row = next(row for row in result["experiments"] if row["task"] == "mnist" and row["seed"] == 7 and row["representation"] == representation)
            probe = Path(row["checkpoint_probe"]["path"])
            probe = probe if probe.is_absolute() else root / probe
            arrays = load_numeric_npz(probe, expected_keys={"inputs", "predictions"})
            inputs = torch.from_numpy(arrays["inputs"])
            model = build_checkpoint_model(row, root)
            with torch.no_grad(), autocast_context(representation, "cpu"):
                prediction = model(inputs).float()
                expected = torch.from_numpy(arrays["predictions"])
                # Library versions can change floating point accumulation order.
                close = bool(torch.allclose(prediction, expected, rtol=1e-4, atol=1e-5))
                if not close:
                    raise RuntimeError("Installed checkpoint inference differs outside the recorded tolerance")
                model(inputs)
                started = time.perf_counter()
                for _ in range(10):
                    model(inputs)
                elapsed = (time.perf_counter() - started) / 10
            inference.append({"record_id": row["id"], "device": "cpu", "examples": len(inputs),
                              "batch_ms": elapsed * 1000, "repeats": 10, "relative_tolerance": 1e-4,
                              "absolute_tolerance": 1e-5, "maximum_delta": float((prediction - expected).abs().max()),
                              "prediction_close": close, "checkpoint_sha256": row["checkpoint_probe"]["checkpoint_sha256"],
                              "execution": "FP32 CPU kernels; chroma8 uses floating point fake quantization"})
            if representation == "fp32":
                integer = convert_integer_cpu(model)
                with torch.no_grad():
                    output = integer(inputs)
                if not torch.isfinite(output).all():
                    raise RuntimeError("Installed integer CPU reference failed")
                inference.append({"source_record_id": row["id"], "device": "cpu", "examples": len(inputs),
                                  "finite": True, "execution": "INT8 weights and INT32 products, FP32 scale/bias/activation"})
    report = {"schema_version": 1, "python": platform.python_version(), "prefix": "<environment>",
              "isolated": sys.flags.isolated == 1, "user_site_enabled": site.ENABLE_USER_SITE,
              "modules": modules, "installed_assets": assets,
              "dependencies": {name: metadata.version(name) for name in ("chromatensor", "torch", "torchvision", "numpy", "matplotlib", "psutil", "pytest")},
              "checks": {"isolated_imports": True, "cube_inverse": inverse,
                         "holographic_decode": finite, "bundled_assets": bundled, "rgbd_tokens": rgbd_ok},
              "rgbd_encoding": rgbd_encoding,
              "deployed_inference": inference,
              "scope": "Clean virtual environment with independently installed dependencies; isolated installed wheel imports and CPU inference"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"python": report["python"], "checks": report["checks"], "inference_records": len(inference)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
