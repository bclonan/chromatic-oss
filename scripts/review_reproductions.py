"""Replay the original review inputs and record whether each defect remains."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import urllib.request
import urllib.error
from unittest.mock import patch
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from chromatensor import ChromaTensor, HierarchicalCodec, LearnedCodec, RGBCodec, UniformCodec
from chromatensor.codecs import codec_from_config
from chromatensor.autograd import ChromaQuantizer
from chromatensor.nn import ChromaAttention, convert_to_chroma
from chromatensor.operators import direct_grid_add
from dashboard.server import DashboardServer
from experiments.common import write_json
from experiments.provenance import capture_provenance
from experiments.cube import train_cube_reconstructor
from experiments.datasets import stratified_indices
from experiments.training import model_for


def causal_probe():
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 10, 20, 30]])
    deltas = {}
    with torch.no_grad():
        for representation in ("fp32", "int8", "chroma8", "fixed24", "rgb24"):
            model = model_for("language", representation, (32, 65), 7, "cpu").eval()
            deltas[representation] = float((model(first)[:, :3] - model(second)[:, :3]).abs().max())
        torch.manual_seed(19)
        attention = ChromaAttention(8, 2, representation="chroma8").eval()
        x = torch.randn(1, 6, 8)
        changed = x.clone()
        changed[:, 3:] *= 100
        attention_delta = float((attention(x, is_causal=True)[:, :3] - attention(changed, is_causal=True)[:, :3]).abs().max())
    return {"reproduced": deltas["chroma8"] > 0 and deltas["fp32"] == 0,
            "seed": 7, "prefix_logits_max_delta": deltas,
            "attention_seed": 19, "attention_prefix_max_delta": attention_delta}


def slicing_probe():
    x = torch.arange(6.).reshape(2, 3)
    actual = ChromaTensor.from_tensor(x)[True, ..., 0].decode()
    expected = x[True, ..., 0]
    return {"reproduced": not torch.equal(actual, expected),
            "expected": expected.tolist(), "actual": actual.tolist()}


def conversion_probe():
    class SharedParameterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.alias = nn.Parameter(torch.randn(3, 3))
            self.layer = nn.Linear(3, 3, bias=False)
            self.layer.weight = self.alias

        def forward(self, x):
            return self.layer(x) + torch.nn.functional.linear(x, self.alias)

    torch.manual_seed(7)
    source = SharedParameterModel()
    converted = convert_to_chroma(source)
    restored = converted.to_torch()
    tie = {"original_tied": source.alias is source.layer.weight,
           "converted_tied": converted.model.alias is converted.model.layer.weight,
           "restored_tied": restored.alias is restored.layer.weight,
           "parameter_counts": [len(list(m.parameters())) for m in (source, converted, restored)]}
    original_attention = nn.MultiheadAttention(8, 2, batch_first=True).eval()
    try:
        converted_attention = convert_to_chroma(original_attention, bits=1)
    except (TypeError, ValueError) as error:
        attention = {"reproduced": False, "unsupported_conversion_rejected": True, "error": str(error)}
    else:
        x = torch.randn(2, 5, 8)
        with torch.no_grad():
            original_output = original_attention(x, x, x)[0]
            converted_output = converted_attention(x, x, x)[0]
        calls = sum(m.forward_calls for m in converted_attention.modules() if isinstance(m, ChromaQuantizer))
        attention = {"reproduced": calls == 0, "quantizer_calls": calls, "bits": 1,
                     "outputs_exact": torch.equal(original_output, converted_output)}
    return {"parameter_tie": {"reproduced": not tie["converted_tied"], **tie},
            "attention_bypass": attention}


def codec_probe():
    codec = LearnedCodec(size=4)
    codec.codebook.double()
    with torch.no_grad():
        codec.codebook.values.copy_(torch.tensor([-1., -.333333333333, .123456789012, 1.], dtype=torch.float64))
    ids = torch.tensor([[1], [2]])
    recovered = codec_from_config(codec.to_config())
    difference = float((codec.decode(ids) - recovered.decode(ids)).abs().max().detach())
    try:
        tiny = LearnedCodec(16, minimum=1e-50, maximum=2e-50)
    except ValueError as error:
        domain = {"reproduced": False, "unsupported_domain_rejected": True, "error": str(error)}
    else:
        x = torch.linspace(1e-50, 2e-50, 16, dtype=torch.float64)
        tiny_decoded = tiny.decode(tiny.encode(x))
        domain = {"reproduced": bool((tiny_decoded.double() < tiny.minimum).all()),
                  "centroid_unique_values": tiny.codebook.values.unique().tolist(),
                  "max_error": float((tiny_decoded - x).abs().max().detach())}
    hierarchy = {}
    rejected = {}
    for levels in (1, 2, 3, 4, 32):
        try:
            residual = HierarchicalCodec(levels=levels, vocabulary_size=2)
        except ValueError as error:
            rejected[str(levels)] = str(error)
            continue
        values = torch.tensor([-1., 0., 1.], dtype=torch.float64)
        hierarchy[str(levels)] = residual.decode(residual.encode(values)).tolist()
    return {"config_precision": {"reproduced": difference > 0, "max_delta": difference,
                                 "original_dtype": str(codec.codebook.values.dtype),
                                 "restored_dtype": str(recovered.codebook.values.dtype)},
            "scalar_domain": domain,
            "binary_hierarchy": {"reproduced": "2" in hierarchy and hierarchy["2"][-1] != 1.,
                                 "decoded_by_level_count": hierarchy, "rejected_multistage": rejected}}


def full_mnist_probe():
    from torchvision.datasets import MNIST
    rows = []
    try:
        for training, count in ((True, 60000), (False, 10000)):
            dataset = MNIST(ROOT / "data", train=training, download=False)
            indices = stratified_indices(dataset.targets, count, 7)
            rows.append({"split": "train" if training else "test", "official_count": len(dataset),
                         "requested_count": count, "selected_count": int(indices.numel())})
    except RuntimeError as error:
        return {"reproduced": None, "reason": str(error), "rows": rows}
    return {"reproduced": any(row["selected_count"] != row["official_count"] for row in rows),
            "provenance": "cached official torchvision MNIST, download=False", "seed": 7, "rows": rows}


def package_probe():
    result = {}
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    for name, path in (("retired_0_1", ROOT / "artifacts/review/retired-packages/chromatensor-0.1.0-py3-none-any.whl"),
                       ("active_release", ROOT / f"artifacts/packages/chromatensor-{version}-py3-none-any.whl")):
        if not path.exists():
            continue
        required = ["chromatensor/cube.py", "chromatensor/holographic.py", "experiments/cube.py",
                    "dashboard/cube_api.py", "dashboard/cube.js", "dashboard/cube.css"]
        with zipfile.ZipFile(path) as archive:
            result[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "missing_files": [item for item in required if item not in archive.namelist()],
                            "exports_cube": b"from .cube import ChromaCube" in archive.read("chromatensor/__init__.py")}
    active = result.get("active_release")
    return {"reproduced": active is None or bool(active["missing_files"]) or not active["exports_cube"],
            "active_version": version, "wheels": result}


def numerical_edge_probe():
    codec = UniformCodec(bits=8, minimum=0, maximum=1)
    signed = ChromaTensor(torch.tensor([[100]], dtype=torch.int8), codec)
    try:
        output = direct_grid_add(signed, signed)
        container = {"reproduced": False, "actual_code": output.codes.tolist()}
    except ValueError as error:
        container = {"reproduced": True, "expected_code": 200, "error": str(error)}
    low, high = 1e20, math.nextafter(1e20, math.inf)
    try:
        logarithmic = RGBCodec(low, high, logarithmic=True)
    except ValueError as error:
        log_result = {"reproduced": False, "degenerate_domain_rejected": True,
                      "bounds": [low, high], "error": str(error)}
    else:
        codes = logarithmic.encode(torch.tensor([low, high], dtype=torch.float64))
        log_result = {"reproduced": torch.equal(codes[0], codes[1]), "bounds": [low, high],
                      "codes": codes.tolist()}
    return {"grid_container": container,
            "logarithmic_domain": log_result}


def host_probe():
    server = DashboardServer(("127.0.0.1", 0), ROOT / "results/latest.json")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        headers = {"Host": "attacker.example", "Origin": "http://attacker.example"}
        get = urllib.request.Request(base + "/api/results", headers=headers)
        try:
            with urllib.request.urlopen(get) as response:
                read_status = response.status
        except urllib.error.HTTPError as error:
            read_status = error.code
        payload = json.dumps({"shape": [2], "values": [0., 1.], "codec": "rgb"}).encode()
        post = urllib.request.Request(base + "/api/tensor", data=payload,
                                     headers={**headers, "Content-Type": "application/json"})
        actual_codes = None
        try:
            with urllib.request.urlopen(post) as response:
                compute_status = response.status
                actual_codes = json.load(response)["codes"]
        except urllib.error.HTTPError as error:
            compute_status = error.code
        return {"reproduced": read_status == 200 and compute_status == 200,
                "foreign_host_get_status": read_status, "foreign_origin_post_status": compute_status,
                "encoded_codes": actual_codes, "scope": "server-side header test; no browser exploitation test"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def artifact_probe():
    with tempfile.TemporaryDirectory(prefix="chromatensor-review-") as directory:
        first = train_cube_reconstructor("direct_rgb24", 7, Path(directory), epochs=2)
        path = Path(first["artifact_path"])
        first_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        second = train_cube_reconstructor("direct_rgb24", 7, Path(directory), epochs=1)
        preserved_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        second_hash = hashlib.sha256(Path(second["artifact_path"]).read_bytes()).hexdigest()
        return {"reproduced": first["artifact_path"] == second["artifact_path"] and first_hash != second_hash,
                "same_path": first["artifact_path"] == second["artifact_path"],
                "first_checkpoint_preserved": first_hash == preserved_hash,
                "checkpoint_hashes": [first_hash, second_hash],
                "validation_mse": [r["metrics"]["validation_loss"] for r in (first, second)],
                "checkpoint_bytes": [r["metrics"]["checkpoint_bytes"] for r in (first, second)],
                "scope": "temporary directory; original cube records and checkpoints untouched"}


def acceptance_probe():
    from scripts import verify
    captured = {}
    def capture(path, value):
        captured.update(value)
    with tempfile.TemporaryDirectory(prefix="chromatensor-skipped-") as directory:
        root = Path(directory)
        (root / "tests").mkdir()
        (root / "tests/test_required.py").write_text("def test_required(): pass\n")
        xml = root / "tests.xml"
        xml.write_text('<testsuites><testsuite tests="1" skipped="1" failures="0" errors="0"><testcase classname="test_required" name="test_required"><skipped/></testcase></testsuite></testsuites>')
        with patch.object(verify, "write_json", side_effect=capture), redirect_stdout(io.StringIO()):
            exit_code = verify.verify(root=root, tests_path=xml)
    return {"reproduced": exit_code == 0 and captured["complete"],
            "provided_tests": 1, "provided_skipped": 1, "exit_code": exit_code,
            "reported_complete": captured["complete"],
            "scope": "temporary workspace with skipped test XML and missing required evidence; canonical acceptance untouched; dedicated negative tests isolate each gate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/repair-regressions.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    findings = {"causal": causal_probe(), "slicing": slicing_probe(), **conversion_probe(),
                **codec_probe(), **numerical_edge_probe(), "full_mnist": full_mnist_probe(), "package": package_probe(),
                "host": host_probe(), "cube_artifacts": artifact_probe(), "acceptance": acceptance_probe()}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {"schema_version": 1, "reviewed_at": datetime.now(timezone.utc).isoformat(),
              "source_revision": revision, "python": sys.version, "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "device": "cpu", "threads": 1,
              "provenance": capture_provenance({"command": "scripts/review_reproductions.py"}, ROOT),
              "reproduction_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "findings": findings}
    write_json(args.output, report)
    for name, value in findings.items():
        print(f"{name}: reproduced={value['reproduced']}")
    print(f"Saved review evidence to {args.output}")
    return 1 if any(row["reproduced"] is not False for row in findings.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
