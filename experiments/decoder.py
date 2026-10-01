"""Train a genuine learned RGB representation and a neural palette-decoder control."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from chromatensor.codecs import LearnedCodec, LearnedRGBCodec, RGBCodec, UniformCodec
from chromatensor.decoders import LearnedNeuralDecoder
from .common import PeakMemory, dataset_hash, error_metrics, seed_all, sync, timed
from .training import tensor_sample


def run_decoder_experiment(config: dict, artifacts=None):
    """Return one measured experiment record, inspector samples and learned manifold.

    Numerical train and validation observations are independent. The auxiliary
    palette decoder trains on 80% of token IDs and reports held-out ID coverage.
    It is a control for reconstruction from fixed RGB labels, not the learned
    color encoder used by the primary experiment.
    """
    seed = int(config.get("seeds", [7])[0])
    device = config.get("device", "cpu")
    steps = int(config.get("decoder_steps", 60 if config.get("profile") == "smoke" else 300))
    train_count = int(config.get("decoder_train_samples", 1024 if config.get("profile") == "smoke" else 8192))
    validation_count = int(config.get("decoder_validation_samples", 512 if config.get("profile") == "smoke" else 4096))
    if train_count < 2 or validation_count < 2:
        raise ValueError("Decoder experiments require at least two train and validation samples")
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    train = (torch.randn(train_count, generator=generator) * 0.35).clamp(-1, 1)
    validation = (torch.randn(validation_count, generator=generator) * 0.35).clamp(-1, 1)
    train_device, validation_device = train.to(device), validation.to(device)
    codec = LearnedRGBCodec(hidden=32, seed=seed).to(device)
    initial = codec.decode(codec.encode(validation_device)).cpu()
    initial_metrics = error_metrics(validation, initial)
    sync(device)
    with PeakMemory(device) as memory:
        start = time.perf_counter()
        codec.fit(train_device, iterations=steps, seed=seed, batch_size=256,
                  validation_samples=validation_device, smoothness_weight=0.001)
        sync(device)
        training_seconds = time.perf_counter() - start
    codes, encode_seconds = timed(lambda: codec.encode(validation_device), device, repeats=5)
    decoded, decode_seconds = timed(lambda: codec.decode(codes), device, repeats=5)
    errors = error_metrics(validation_device, decoded)
    reconstructed_train = codec.decode(codec.encode(train_device))
    train_mse = float((reconstructed_train - train_device).square().mean())
    master_bytes = sum(parameter.numel() * parameter.element_size() for parameter in codec.parameters())
    metadata_bytes = len(json.dumps(codec.to_config(), allow_nan=False).encode())

    # Fit a scalar palette only on numerical training data. The neural decoder
    # then omits every fifth palette ID to make unseen-token behavior explicit.
    book = LearnedCodec(size=256).fit(train, iterations=20, seed=seed)
    all_ids = torch.arange(book.size)
    seen = (all_ids % 5) != 0
    neural_decoder = LearnedNeuralDecoder(hidden=64, seed=seed).to(device)
    palette_rgb = book.colors(all_ids.unsqueeze(-1)).to(device)
    palette_targets = book.decode(all_ids.unsqueeze(-1)).detach().to(device)
    neural_decoder.fit(palette_rgb[seen.to(device)], palette_targets[seen.to(device)], steps=steps, seed=seed,
                       validation_rgb=palette_rgb[(~seen).to(device)],
                       validation_targets=palette_targets[(~seen).to(device)])
    with torch.no_grad():
        validation_ids = book.encode(validation)[..., 0].long()
        palette_prediction = neural_decoder(book.colors(validation_ids.unsqueeze(-1)).to(device)).cpu()
        unseen_rgb = palette_rgb[(~seen).to(device)]
        unseen_targets = palette_targets[(~seen).to(device)]
        unseen_prediction = neural_decoder(unseen_rgb)
        exact_lookup = book.decode(validation_ids.unsqueeze(-1)).detach()
        validation_unseen = (~seen)[validation_ids]
    palette_control = {
        "method": "RGB MLP decoding of a separately fitted scalar codebook",
        "token_training_count": int(seen.sum()), "heldout_token_count": int((~seen).sum()),
        "validation_values_with_unseen_token": int(validation_unseen.sum()),
        "validation_unseen_token_fraction": float(validation_unseen.float().mean()),
        "numerical_heldout_errors": error_metrics(validation, palette_prediction),
        "unseen_centroid_errors": error_metrics(unseen_targets, unseen_prediction),
        "exact_lookup_errors": error_metrics(validation, exact_lookup),
        "curves": neural_decoder.fit_history,
        "interpretation_scope": "held-out numerical observations and omitted token IDs; fixed RGB labels are not jointly learned here",
    }

    checkpoint = None
    if artifacts is not None:
        root = Path(artifacts)
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / f"neural_rgb_s{seed}.npz"
        np.savez_compressed(checkpoint, **{name: value.detach().cpu().numpy() for name, value in codec.state_dict().items()})
        checkpoint.with_suffix(".json").write_text(json.dumps({"minimum": codec.minimum, "maximum": codec.maximum,
                                                               "hidden": codec.hidden, "seed": seed,
                                                               "fit_metadata": codec.fit_metadata}, indent=2), encoding="utf-8")
        np.savez_compressed(root / f"neural_palette_decoder_s{seed}.npz",
                            **{name: value.detach().cpu().numpy() for name, value in neural_decoder.state_dict().items()})
        (root / f"neural_palette_decoder_s{seed}.json").write_text(json.dumps({"minimum": -1, "maximum": 1,
                                                                             "hidden": 64, "seed": seed,
                                                                             "trained_token_ids": all_ids[seen].tolist()}, indent=2), encoding="utf-8")
    with torch.no_grad():
        controls = {"rgb24": error_metrics(validation, RGBCodec().decode(RGBCodec().encode(validation))),
                    "uniform8": error_metrics(validation, UniformCodec(8).decode(UniformCodec(8).encode(validation))),
                    "uniform24": error_metrics(validation, UniformCodec(24).decode(UniformCodec(24).encode(validation)))}
    metrics = {"training_loss": train_mse, "validation_loss": errors["mse"],
               "initial_validation_loss": initial_metrics["mse"], "encoding_error": errors["mae"],
               "reconstruction_error": errors["mae"], "bits_per_value": 24,
               "bytes_model": master_bytes, "codec_metadata_bytes": metadata_bytes,
               "training_seconds": training_seconds, "training_throughput": steps * min(256, train_count) / training_seconds,
               "inference_seconds": encode_seconds + decode_seconds,
               "inference_throughput": validation_count / (encode_seconds + decode_seconds),
               "encode_ms": encode_seconds * 1000, "decode_ms": decode_seconds * 1000,
               **errors, **memory.report()}
    identifier = f"decoder-learned_rgb-s{seed}"
    record = {"id": identifier, "task": "decoder", "representation": "learned_rgb", "seed": seed,
              "device": device, "metrics": metrics,
              "curves": [{"epoch": row["step"], "train_loss": row["training_loss"], "val_loss": row["validation_loss"]}
                         for row in codec.fit_history],
              "dataset": {"name": "independent clipped normal scalar samples std=0.35", "training_samples": train_count,
                          "validation_samples": validation_count, "sha256": dataset_hash(train, validation),
                          "seed": seed, "encoder_fit_scope": "training observations only; validation does not update weights"},
              "execution": {"fully_chromatic": False, "encoder": "FP32 MLP 1->32->32->3 sigmoid",
                            "forward_representation": "three learned uint8 RGB channels",
                            "decoder": "FP32 MLP 3->32->32->1 tanh",
                            "gradient_method": "identity straight-through byte quantization",
                            "dense_and_optimizer_dtype": "torch.float32"},
              "artifact_path": None if checkpoint is None else str(checkpoint),
              "model_config": {"schema_version": 1, "task": "decoder", "representation": "learned_rgb",
                               "seed": seed, "minimum": codec.minimum, "maximum": codec.maximum,
                               "hidden": codec.hidden},
              "palette_decoder_control": palette_control, "equal_data_controls": controls,
              "non_claims": ["A learned RGB representation is not automatically better than fixed-point quantization.",
                             "The 24-bit payload excludes neural model weights and per-file metadata.",
                             "Decoder fitting does not remove floating-point kernels."]}
    probes = torch.linspace(-1, 1, 256)
    probe_rgb = codec.encode(probes).cpu()
    probe_decoded = codec.decode(probe_rgb).cpu()
    nearest_probe = torch.round((train + 1) / 2 * 255).long().clamp(0, 255)
    manifold = {"name": "jointly_learned_rgb_manifold", "values": probes.tolist(), "decoded_values": probe_decoded.tolist(),
                "rgb": probe_rgb.tolist(), "frequencies": torch.bincount(nearest_probe, minlength=256).tolist(),
                "frequency_basis": "training scalar values assigned to nearest evenly spaced probe, not distinct RGB vocabulary IDs",
                "fit_loss": [row["training_loss"] for row in codec.fit_history], "fit_samples": train_count,
                "fit_distribution": "clipped normal std=0.35, training split only",
                "unique_probe_colors": int(torch.unique(probe_rgb, dim=0).shape[0]),
                "mean_adjacent_rgb_distance": float(torch.linalg.vector_norm(probe_rgb[1:].float() - probe_rgb[:-1].float(), dim=-1).mean()),
                "objective": "joint neural encode/decode MSE with byte STE and RGB neighbor smoothness penalty"}
    samples = [tensor_sample("jointly learned RGB held-out values", validation[:256], codec, identifier)]
    if checkpoint is not None:
        record["metrics"]["checkpoint_bytes"] = checkpoint.stat().st_size
        from .checkpoint_probe import write_checkpoint_probe
        write_checkpoint_probe(record, validation)
    return record, samples, manifold

