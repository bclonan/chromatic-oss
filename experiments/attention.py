"""Train paired ordinary and codebook attention on analytic phase sequences."""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import math
from pathlib import Path
import time

import torch
from torch import nn

from chromatensor import RGBDCodec, save_chroma_checkpoint
from chromatensor.nn import ChromaAttention, module_quantization_report
from .common import dataset_hash, environment, PeakMemory, seed_all, sync, timed, write_json
from .training import tensor_sample


MODES = ("fp32_attention_dot", "chroma8_attention_dot", "chroma8_attention_codebook")


def phase_sequence_data(seed: int, device: str = "cpu", *, count: int = 512,
                        validation_count: int = 128, sequence_length: int = 12):
    """Predict the next unit-circle point at each position from its causal history."""
    generator = torch.Generator().manual_seed(seed + 7000)
    total = count + validation_count
    initial = torch.rand(total, 1, generator=generator) * (2 * math.pi)
    velocity = .08 + .40 * torch.rand(total, 1, generator=generator)
    time_steps = torch.arange(sequence_length + 1).float().unsqueeze(0)
    phase = initial + velocity * time_steps
    # Vary speed smoothly; the task has a real circular target and causal context.
    phase = phase + .12 * torch.sin(phase * 2)
    coordinates = torch.stack((phase.cos(), phase.sin()), -1)
    x, y = coordinates[:, :-1].contiguous(), coordinates[:, 1:].contiguous()
    provenance = {
        "source": "analytic variable-speed phase sequences",
        "formula": "theta[t] = theta0 + omega*t + 0.12*sin(2*(theta0+omega*t))",
        "target": "cos(theta[t+1]), sin(theta[t+1]) at every causal sequence position",
        "train_examples": count, "validation_examples": validation_count,
        "sequence_length": sequence_length, "selection_seed": seed + 7000,
        "sha256": dataset_hash(x, y),
        "note": "Targets and sin/cos input features use FP32; no text corpus or image dataset is involved.",
    }
    return x[:count].to(device), y[:count].to(device), x[count:].to(device), y[count:].to(device), provenance


class PhaseAttentionPredictor(nn.Module):
    def __init__(self, mode: str, *, hidden: int = 24, heads: int = 3,
                 sequence_length: int = 12, num_codes: int = 32):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown attention experiment mode {mode!r}")
        # Build every shared parameter before constructing the experimental layer.
        # Conversion then copies all projection values from this exact control.
        self.input_projection = nn.Linear(2, hidden)
        control = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.Tanh(), nn.Linear(hidden * 2, 2))
        positions = torch.arange(sequence_length).float().unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.) / hidden))
        position = torch.zeros(sequence_length, hidden)
        position[:, 0::2] = torch.sin(positions * frequencies)
        position[:, 1::2] = torch.cos(positions * frequencies[:position[:, 1::2].shape[1]])
        self.register_buffer("position", position * .2)
        self.attention = ChromaAttention.from_torch(
            control, representation="chroma8", quantize_weights=mode != MODES[0],
            quantize_activations=mode != MODES[0],
            similarity_mode="codebook_similarity" if mode == MODES[2] else "dot_product",
            num_codes=num_codes,
        )
        self.mode = mode

    def embedded(self, x):
        return self.input_projection(x) + self.position[:x.shape[1]]

    def forward(self, x):
        embedded = self.embedded(x)
        return self.output(self.norm(embedded + self.attention(embedded, is_causal=True)))

    def auxiliary_loss(self):
        return self.attention.auxiliary_loss()

    def calibrate_codebook(self, train_inputs):
        """Initialize codes from training Q/K vectors, without touching held-out data."""
        if self.attention.codebook is None:
            return
        with torch.no_grad():
            embedded = self.embedded(train_inputs[:64])
            q = self.attention._split(self.attention.q_proj(embedded)).reshape(-1, self.attention.head_dim)
            k = self.attention._split(self.attention.k_proj(embedded)).reshape(-1, self.attention.head_dim)
            candidates = torch.cat((q, k))
            indices = torch.linspace(0, len(candidates) - 1, self.attention.codebook.num_codes,
                                     device=candidates.device).long()
            self.attention.codebook.codebook.copy_(candidates[indices])


def _shared_initialization_hash(model):
    values = [value.detach().cpu() for name, value in model.named_parameters()
              if "codebook" not in name]
    return dataset_hash(*values)


def _evaluate(model, x, target):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        loss = float((prediction - target).square().mean())
        difference = torch.atan2(prediction[..., 1], prediction[..., 0]) - torch.atan2(target[..., 1], target[..., 0])
        angle = torch.remainder(difference + math.pi, 2 * math.pi) - math.pi
        return {"validation_loss": loss, "angular_mae_radians": float(angle.abs().mean())}


def run_attention_ablation(config, artifact_dir=None, *, seeds=None):
    device = str(config.get("device", "cpu"))
    seeds = list(seeds if seeds is not None else config.get("seeds", [0, 1]))
    artifact_dir = Path(artifact_dir or config.get("artifact_dir", "artifacts/attention"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    count = int(config.get("attention_train_count", 512))
    validation_count = int(config.get("attention_validation_count", 128))
    epochs = int(config.get("attention_epochs", 32))
    length = int(config.get("attention_sequence_length", 12))
    hidden = int(config.get("attention_hidden", 24))
    heads = int(config.get("attention_heads", 3))
    num_codes = int(config.get("attention_num_codes", 32))
    if min(count, validation_count, epochs, length) < 1:
        raise ValueError("Attention data counts, epochs, and sequence length must be positive")
    records, samples = [], []
    for seed in seeds:
        tx, ty, vx, vy, provenance = phase_sequence_data(seed, device, count=count,
            validation_count=validation_count, sequence_length=length)
        hashes = []
        for mode in MODES:
            seed_all(seed)
            model = PhaseAttentionPredictor(mode, hidden=hidden, heads=heads,
                sequence_length=length, num_codes=num_codes).to(device)
            initial_hash = _shared_initialization_hash(model)
            hashes.append(initial_hash)
            model.calibrate_codebook(tx)
            optimizer = torch.optim.Adam(model.parameters(), lr=.004)
            generator = torch.Generator(device=device).manual_seed(seed + 1000)
            initial = _evaluate(model, vx, vy)
            curves = [{"epoch": 0, "train_loss": None, "val_loss": initial["validation_loss"],
                       "metric": initial["angular_mae_radians"]}]
            elapsed = 0.
            with PeakMemory(device) as memory:
                for epoch in range(1, epochs + 1):
                    model.train()
                    permutation = torch.randperm(len(tx), generator=generator, device=device)
                    task_total, auxiliary_total = 0., 0.
                    sync(device)
                    started = time.perf_counter()
                    for offset in range(0, len(tx), 64):
                        ids = permutation[offset:offset + 64]
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(tx[ids])
                        task_loss = (prediction - ty[ids]).square().mean()
                        auxiliary = model.auxiliary_loss()
                        loss = task_loss + .1 * auxiliary
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite attention training loss in {mode}/{seed}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        task_total += float(task_loss.detach()) * len(ids)
                        auxiliary_total += float(auxiliary.detach()) * len(ids)
                    sync(device)
                    elapsed += time.perf_counter() - started
                    measured = _evaluate(model, vx, vy)
                    curves.append({"epoch": epoch, "train_loss": task_total / len(tx),
                        "auxiliary_loss": auxiliary_total / len(tx), "val_loss": measured["validation_loss"],
                        "metric": measured["angular_mae_radians"]})
            model.eval()
            with torch.no_grad():
                _, inference_seconds = timed(lambda current=model: current(vx[:64]), device=device, repeats=5, warmup=1)
            run_id = f"attention-{mode}-s{seed}"
            path = artifact_dir / (run_id + ".ctcheckpoint")
            checkpoint_bytes = save_chroma_checkpoint(model, path, RGBDCodec())
            parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
            execution = module_quantization_report(model)
            execution["quantization_scope"] = "Q, K, V and output attention projections only; input projection, residual, normalization and prediction head use FP32"
            execution["codebook_calibration"] = f"{num_codes} vectors sampled from Q/K on first 64 training examples, then trained" if model.attention.codebook is not None else None
            metrics = {**measured, **memory.report(), "training_loss": curves[-1]["train_loss"],
                "auxiliary_loss": curves[-1]["auxiliary_loss"], "initial_validation_loss": initial["validation_loss"],
                "initial_angular_mae_radians": initial["angular_mae_radians"], "epochs": epochs,
                "learning_rate": .004, "training_seconds": elapsed,
                "training_throughput": count * epochs / elapsed,
                "inference_batch_ms": inference_seconds * 1000,
                "inference_throughput": min(64, validation_count) / inference_seconds,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "live_parameter_bytes": parameter_bytes, "bytes_model": parameter_bytes,
                "checkpoint_bytes": checkpoint_bytes,
                "bits_per_value": 32 if mode == MODES[0] else 8,
                "storage_scope": "actual FP32 master parameters, including learned code vectors; chromatic forward buffers are reported separately"}
            record = {"id": run_id, "task": "attention", "representation": mode, "seed": seed,
                "device": device, "metrics": metrics, "curves": curves, "dataset": provenance,
                "shared_initialization_sha256": initial_hash, "execution": execution,
                "artifact_path": str(path), "evidence_scope": "analytic causal phase-sequence training, no claim about general language modeling",
                "model_config": {"schema_version": 1, "task": "attention", "representation": mode,
                                 "seed": seed, "hidden": hidden, "heads": heads,
                                 "sequence_length": length, "num_codes": num_codes}}
            records.append(record)
            samples.append(tensor_sample(run_id + " attention probabilities", model.attention.last_attention,
                                         experiment_id=run_id))
            if model.attention.codebook is not None:
                bottleneck = model.attention.codebook
                indices = bottleneck.last_indices.detach().cpu().flatten()[:256]
                vectors = bottleneck.codebook.detach().cpu()[indices]
                samples.append({"name": run_id + " learned latent codes", "shape": [16, math.ceil(len(indices) / 16)],
                    "values": vectors.norm(dim=-1).tolist(), "decoded": vectors.norm(dim=-1).tolist(),
                    "rgb": bottleneck.colors.detach().cpu()[indices].tolist(), "tokens": indices.tolist(),
                    "codec": {"name": "learned attention vector codes", "features_per_code": bottleneck.features,
                              "num_codes": bottleneck.num_codes}, "experiment_id": run_id,
                    "note": "Each RGB symbol names a learned vector; scalar display values are vector norms, not scalar codec reconstruction.",
                    "metrics": {"used_codes": bottleneck.usage()["used_codes"], "perplexity": bottleneck.usage()["perplexity"]}})
            from .checkpoint_probe import write_checkpoint_probe
            write_checkpoint_probe(record, vx)
            del model, optimizer
        if len(set(hashes)) != 1:
            raise RuntimeError("Attention shared initialization differed between paired modes")
    return records, [sample for sample in samples if sample is not None]


def run_attention_experiments(config):
    """Convenience API for runners that collect experiment records only."""
    return run_attention_ablation(config)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/attention.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--epochs", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = {"device": args.device, "seeds": args.seeds, "attention_epochs": args.epochs}
    from .provenance import capture_provenance
    provenance = capture_provenance(config)
    records, samples = run_attention_ablation(config)
    write_json(args.output, {"config": config, "environment": environment(), "experiments": records,
                             "tensor_samples": samples, "provenance": provenance})
    for record in records:
        metrics = record["metrics"]
        print(record["id"], "mse", round(metrics["validation_loss"], 6),
              "angular_mae", round(metrics["angular_mae_radians"], 6),
              "initial_mse", round(metrics["initial_validation_loss"], 6), flush=True)


if __name__ == "__main__":
    main()
