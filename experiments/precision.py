from __future__ import annotations

import io
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from chromatensor import ChromaTensor, RGBCodec, RGBDCodec, HSVCodec, HueCodec, UniformCodec, LearnedCodec, HierarchicalCodec, save_ctensor
from chromatensor.codecs import LearnedRGBDCodec
from chromatensor.operators import LookupOperator, LearnedOperator, direct_grid_add, chroma_add
from .common import seed_all, timed, error_metrics
from .training import tensor_sample


def numerical_precision(config, artifacts):
    seed_all(config["seeds"][0])
    count = config["precision_samples"]
    generator = torch.Generator().manual_seed(config["seeds"][0])
    distributions = {"uniform": torch.rand(count, generator=generator, dtype=torch.float64) * 2 - 1, "normal": torch.randn(count, generator=generator, dtype=torch.float64) * 0.2, "laplace": torch.distributions.Laplace(0.0, 0.15).sample((count,)).double(), "bimodal": torch.randn(count, generator=generator, dtype=torch.float64) * 0.08 + torch.randint(2, (count,), generator=generator).double() * 1.2 - 0.6}
    fit_samples = (torch.randn(131072, generator=generator) * 0.25).clamp(-1, 1)
    vocabularies = [16, 64, 256, 1024, 4096, 65536]
    books = [(size, LearnedCodec(size=size).fit(fit_samples, iterations=8, seed=config["seeds"][0])) for size in vocabularies]
    codecs = [("rgb24", RGBCodec()), ("rgbd32", RGBDCodec()), ("hsv24_effective8", HSVCodec()), ("hue8", HueCodec(period=2, origin=-1)), ("int4", UniformCodec(bits=4)), ("int8", UniformCodec(bits=8)), ("fixed24", UniformCodec(bits=24)), ("residual16_depth3", HierarchicalCodec(levels=3)), ("adaptive_residual", HierarchicalCodec(levels=3, tolerance=0.002)), ("learned_rgbd", LearnedRGBDCodec().fit(fit_samples, iterations=8))]
    codecs += [(f"learned_vocab{size}", book) for size, book in books]
    # Uniform and learned codebooks receive identical code budgets.
    codecs += [(f"uniform_vocab{size}", UniformCodec(bits=int(math.log2(size)))) for size in vocabularies if size not in (16, 256)]
    rows, samples = [], []
    for distribution, raw in distributions.items():
        # Compare bounded codecs on the same clipped samples. Keep clipping
        # counts visible rather than attributing range truncation to quantization.
        values = raw.clamp(-1, 1)
        clipped = int((raw != values).sum())
        for name, codec in codecs:
            start = time.perf_counter()
            codes = codec.encode(values)
            encode_seconds = time.perf_counter() - start
            start = time.perf_counter()
            decoded = codec.decode(codes).detach().double()
            decode_seconds = time.perf_counter() - start
            circular = 2 if name == "hue8" else None
            errors = error_metrics(values, decoded, circular)
            original_hist = torch.histc(values.float(), bins=64, min=-1, max=1).double() + 1
            decoded_hist = torch.histc(decoded.float(), bins=64, min=-1, max=1).double() + 1
            p, q = original_hist / original_hist.sum(), decoded_hist / decoded_hist.sum()
            byte_count = codes.numel() * codes.element_size()
            metadata = len(json.dumps(codec.to_config()).encode())
            row = {"distribution": distribution, "codec": name, "samples": count, "clipped_samples": clipped, **errors, "kl_histogram": float((p * (p / q).log()).sum()), "bits_per_value": byte_count * 8 / count, "nominal_bits_per_value": codec.bits_per_value, "storage_bytes": byte_count, "metadata_bytes": metadata, "total_bytes": byte_count + metadata, "encode_ms": encode_seconds * 1000, "decode_ms": decode_seconds * 1000, "encode_throughput": count / encode_seconds, "decode_throughput": count / decode_seconds, "error_domain": "circular period=2" if circular else "linear", "fit_distribution": "separate clipped normal std=0.25" if name.startswith("learned") else None}
            if isinstance(codec, HierarchicalCodec):
                lengths = codes[..., -1].long()
                row["tokens_per_value"] = float(lengths.double().mean())
                row["packed_active_token_bytes_with_lengths"] = int(lengths.sum()) + lengths.numel()
            rows.append(row)
            if distribution == "uniform" and name in ("rgb24", "hsv24_effective8", "hue8", "adaptive_residual", "learned_rgbd"):
                samples.append(tensor_sample(distribution + " " + name, values[:256], codec))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            decoded, seconds = timed(lambda: values.to(dtype).double(), repeats=2, warmup=0)
            rows.append({"distribution": distribution, "codec": str(dtype).replace("torch.", ""), "samples": count, "clipped_samples": clipped, **error_metrics(values, decoded), "bits_per_value": torch.empty((), dtype=dtype).element_size() * 8, "storage_bytes": count * torch.empty((), dtype=dtype).element_size(), "metadata_bytes": 0, "encode_ms": seconds * 1000, "decode_ms": None, "error_domain": "linear"})
        print(f"Precision {distribution}: {count:,} values x {len(codecs)+3} representations", flush=True)
    codebooks = []
    for size, codec in books:
        ids = codec.encode(fit_samples)[..., 0].long()
        codebooks.append({"name": f"learned_{size}", "values": codec.codebook.values.detach().tolist(), "rgb": codec.codebook.palette.tolist(), "frequencies": torch.bincount(ids, minlength=size).tolist(), "fit_loss": codec.codebook.fit_history, "fit_samples": len(fit_samples), "fit_distribution": "normal std=0.25 clipped [-1,1]", "objective": "scalar Lloyd mean squared error, colors label learned centroids"})
    storage = file_storage_comparison(distributions["normal"][:16384].float(), artifacts)
    return rows, codebooks, samples, storage


def file_storage_comparison(values, artifacts):
    root = Path(artifacts) / "storage"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, codec in [("rgb24", RGBCodec()), ("rgbd32", RGBDCodec()), ("int8", UniformCodec(bits=8)), ("int4", UniformCodec(bits=4)), ("residual", HierarchicalCodec(tolerance=0.002))]:
        chroma = ChromaTensor.from_tensor(values, codec)
        for compress in (False, True):
            path = root / f"{name}_{'zlib' if compress else 'raw'}.ctensor"
            size = save_ctensor(chroma, path, compress=compress)
            rows.append({"format": path.name, "bytes": size, "values": len(values), "bits_per_value_with_metadata": size * 8 / len(values), "reconstruction_mae": error_metrics(values, chroma.decode())["mae"]})
    for dtype in (np.float32, np.float16):
        array = values.numpy().astype(dtype)
        for compressed in (False, True):
            path = root / f"{np.dtype(dtype).name}.{'npz' if compressed else 'npy'}"
            if compressed:
                np.savez_compressed(path, values=array)
            else:
                np.save(path, array, allow_pickle=False)
            size = path.stat().st_size
            rows.append({"format": path.name, "bytes": size, "values": len(values), "bits_per_value_with_metadata": size * 8 / len(values), "reconstruction_mae": error_metrics(values, torch.from_numpy(array).float())["mae"]})
    path = root / "fp32.pt"
    torch.save(values, path)
    rows.append({"format": path.name, "bytes": path.stat().st_size, "values": len(values), "bits_per_value_with_metadata": path.stat().st_size * 8 / len(values), "reconstruction_mae": 0.0})
    # A genuinely packed INT4 numerical control, with separate domain metadata.
    codes = UniformCodec(bits=4).encode(values)[..., 0].numpy()
    packed = (codes[0::2] | (codes[1::2] << 4)).tobytes()
    path = root / "packed_int4.bin"
    path.write_bytes(packed)
    metadata = {"minimum": -1, "maximum": 1, "shape": list(values.shape), "packing": "two 4-bit IDs per byte, low nibble first"}
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
    rows.append({"format": path.name, "bytes": len(packed), "metadata_bytes": metadata_path.stat().st_size, "values": len(values), "bits_per_value_with_metadata": (len(packed) + metadata_path.stat().st_size) * 8 / len(values), "reconstruction_mae": error_metrics(values, UniformCodec(bits=4).decode(torch.from_numpy(codes).unsqueeze(-1)))["mae"]})
    # The equal-budget numerical 24-bit baseline uses exactly three bytes per
    # fixed-point index. Tensor-library int32 padding is not a color advantage.
    numeric_codec = UniformCodec(bits=24)
    numeric_codes = numeric_codec.encode(values)[..., 0].long().numpy()
    packed24 = np.stack([(numeric_codes >> 16) & 255, (numeric_codes >> 8) & 255, numeric_codes & 255], axis=-1).astype(np.uint8).tobytes()
    path = root / "packed_fixed24.bin"
    path.write_bytes(packed24)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"minimum": -1, "maximum": 1, "shape": list(values.shape), "packing": "three bytes big-endian fixed-point index"}, separators=(",", ":")), encoding="utf-8")
    rows.append({"format": path.name, "bytes": len(packed24), "metadata_bytes": metadata_path.stat().st_size, "values": len(values), "bits_per_value_with_metadata": (len(packed24) + metadata_path.stat().st_size) * 8 / len(values), "reconstruction_mae": error_metrics(values, numeric_codec.decode(torch.from_numpy(numeric_codes).unsqueeze(-1)))["mae"]})
    for row in rows:
        row["artifact_path"] = str(root / row["format"])
    return rows


def operator_benchmarks(config):
    seed_all(config["seeds"][0])
    codec = UniformCodec(bits=4, minimum=0, maximum=1)
    a, b = torch.rand(8192), torch.rand(8192)
    ca, cb = ChromaTensor.from_tensor(a, codec), ChromaTensor.from_tensor(b, codec)
    lookup = LookupOperator(codec)
    seed_all(config["seeds"][0])
    learned = LearnedOperator(16)
    losses = learned.fit(lookup, steps=400, seed=config["seeds"][0])
    methods = [("decode_compute_encode", lambda: chroma_add(ca, cb), "FP64 decode, float addition, integer re-encode"), ("lookup", lambda: lookup(ca, cb), "integer table indexing; table built with floats"), ("direct_integer_grid", lambda: direct_grid_add(ca, cb), "integer code addition and saturation"), ("learned", lambda: ChromaTensor(learned(ca.codes[..., 0], cb.codes[..., 0]).to(torch.uint8).unsqueeze(-1), codec), "FP32 embeddings/MLP, integer argmax; no scalar decode")]
    truth = (ca.decode() + cb.decode()).clamp(0, 1)
    rows = []
    for name, function, execution in methods:
        result, seconds = timed(function, repeats=10)
        rows.append({"method": name, **error_metrics(truth, result.decode()), "batch_values": len(a), "latency_ms": seconds * 1000, "throughput": len(a) / seconds, "execution": execution})
    ids = torch.arange(16)
    aa, bb = torch.meshgrid(ids, ids, indexing="ij")
    accuracy = float((learned(aa, bb) == lookup.table[..., 0].long()).float().mean())
    rows[-1].update({"training_loss": losses[-1], "initial_training_loss": losses[0], "table_accuracy": accuracy, "generalization_scope": "all 256 pairs used in training; functional approximation only", "training_steps": len(losses)})
    # Compare repeated table relocation with a cached device table. This is a
    # measured Python reference optimization, not a custom kernel claim.
    table = lookup.table
    _, raw_seconds = timed(lambda: chroma_add(ca, cb), repeats=20)
    _, cached_seconds = timed(lambda: table[ca.codes[..., 0].long(), cb.codes[..., 0].long()], repeats=20)
    optimization = {"target": "repeated finite addition", "baseline_ms": raw_seconds * 1000, "cached_table_ms": cached_seconds * 1000, "speedup": raw_seconds / cached_seconds, "scope": "table gather versus decode/compute/encode, excluding output wrapper validation; CPU Python/PyTorch, no custom kernel"}
    return rows, optimization
