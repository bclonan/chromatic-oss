from __future__ import annotations

import gc
import io
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from chromatensor import ChromaTensor, RGBCodec, RGBDCodec, UniformCodec, HueCodec, save_chroma_checkpoint
from chromatensor.nn import convert_to_chroma, module_quantization_report, CodebookBottleneck
from .common import seed_all, sync, timed, PeakMemory, error_metrics, autocast_context
from .datasets import scalar_data, image_data, language_data
from .models import StandardMLP, TinyTransformer, convert_integer_cpu

REPRESENTATIONS = ["fp32", "fp16", "bf16", "int8", "int4", "fixed24", "chroma8", "rgb24"]


def model_for(task, representation, dimensions, seed, device):
    seed_all(seed)
    inputs, outputs = dimensions
    model = TinyTransformer(outputs) if task == "language" else StandardMLP(inputs, 128 if task in ("mnist", "cifar") else 64, outputs)
    if task == "periodic":
        return model.to(device)
    if representation == "latent":
        model.layers.insert(2, CodebookBottleneck(64, 32))
    elif representation not in ("fp32", "fp16", "bf16"):
        bits = 24 if representation in ("rgb24", "fixed24") else 4 if representation == "int4" else 8
        quant_rep = "chroma8" if representation.startswith("chroma") else representation
        model = convert_to_chroma(model, bits=bits, representation=quant_rep, quantize_weights=representation != "chroma_activations", quantize_activations=representation != "chroma_weights")
    # Model construction and conversion may consume random numbers. Batches get
    # their own generator, with the same seed across each paired representation.
    return model.to(device)


def task_loss(task, prediction, target):
    if task in ("scalar", "periodic"):
        return nn.functional.mse_loss(prediction.float(), target.float())
    return nn.functional.cross_entropy(prediction.float().reshape(-1, prediction.shape[-1]), target.reshape(-1))


def auxiliary_loss(model):
    values = [module.auxiliary_loss() for module in model.modules() if isinstance(module, CodebookBottleneck)]
    return sum(values) if values else 0.0


def evaluate(model, x, y, task, representation, device, batch_size=256):
    model.eval()
    total_loss, correct, amount, angular = 0.0, 0.0, 0, 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            bx, by = x[start:start + batch_size], y[start:start + batch_size]
            with autocast_context(representation, device):
                output = model(bx)
            loss = task_loss(task, output, by)
            count = by.numel() if task == "language" else len(bx)
            total_loss += float(loss) * count
            amount += count
            if task in ("scalar", "periodic"):
                if task == "periodic":
                    delta = torch.atan2(output[:, 1], output[:, 0]) - torch.atan2(by[:, 1], by[:, 0])
                    angular += float(torch.remainder(delta + torch.pi, 2 * torch.pi).sub(torch.pi).abs().sum())
            else:
                correct += float((output.argmax(-1) == by).sum())
    average = total_loss / amount
    metrics = {"validation_loss": average}
    if task in ("mnist", "cifar", "language"):
        metrics["accuracy"] = correct / amount
    if task == "language":
        metrics["perplexity"] = math.exp(min(average, 40))
    if task == "periodic":
        metrics["angular_mae_radians"] = angular / amount
    return metrics


def tensor_sample(name, tensor, codec=None, experiment_id=None):
    tensor = tensor.detach().float().cpu().flatten()[:256]
    if not tensor.numel():
        return None
    minimum, maximum = float(tensor.min()), float(tensor.max())
    if minimum == maximum:
        minimum, maximum = minimum - 1, maximum + 1
    codec = codec or UniformCodec(bits=8, minimum=minimum, maximum=maximum)
    chroma = ChromaTensor.from_tensor(tensor, codec)
    return {"name": name, "shape": [16, math.ceil(len(tensor) / 16)], "values": tensor.tolist(), "decoded": chroma.decode().tolist(), "rgb": chroma.rgb.tolist(), "tokens": chroma.codes[..., 0].long().tolist() if chroma.codes.shape[-1] == 1 else chroma.codes.long().tolist(), "codec": codec.to_config(), "experiment_id": experiment_id, "metrics": error_metrics(tensor, chroma.decode())}


def storage_metrics(model, representation, artifact_dir, name):
    floats = [p.detach() for p in model.parameters()]
    count = sum(p.numel() for p in floats)
    master = sum(p.numel() * p.element_size() for p in model.parameters())
    bits = 24 if representation in ("rgb24", "fixed24") else 4 if representation == "int4" else 16 if representation in ("fp16", "bf16") else 32 if representation in ("fp32", "chroma_activations", "latent", "scalar8", "hue8", "numeric_circular8", "raw_sincos64") else 8
    codec_errors, encode_seconds, decode_seconds, code_bytes, metadata_bytes = [], 0.0, 0.0, 0, 0
    for parameter in floats:
        if representation in ("fp16", "bf16"):
            target_dtype = torch.float16 if representation == "fp16" else torch.bfloat16
            casted, enc = timed(lambda: parameter.to(target_dtype), parameter.device, repeats=2, warmup=0)
            decoded, dec = timed(lambda: casted.float(), parameter.device, repeats=2, warmup=0)
            code_bytes += casted.numel() * casted.element_size()
            encode_seconds += enc
            decode_seconds += dec
            codec_errors.append((error_metrics(parameter, decoded)["mae"], parameter.numel()))
            continue
        lo, hi = float(parameter.min()), float(parameter.max())
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        codec = RGBCodec(lo, hi) if representation == "rgb24" else RGBDCodec() if representation == "fp32" else UniformCodec(bits=bits, minimum=lo, maximum=hi)
        # The measured diagnostic code storage is distinct from an executable
        # frozen inference model and from live FP32 optimizer/master weights.
        chroma, enc = timed(lambda: ChromaTensor.from_tensor(parameter, codec), parameter.device, repeats=2, warmup=0)
        decoded, dec = timed(chroma.decode, parameter.device, repeats=2, warmup=0)
        encode_seconds += enc
        decode_seconds += dec
        code_bytes += chroma.codes.numel() * chroma.codes.element_size()
        metadata_bytes += len(str(codec.to_config()).encode())
        codec_errors.append((error_metrics(parameter, decoded)["mae"], parameter.numel()))
    path = Path(artifact_dir) / (name + ".ctcheckpoint")
    checkpoint_bytes = save_chroma_checkpoint(model, path, RGBDCodec())
    torch_buffer = io.BytesIO()
    torch.save(model.state_dict(), torch_buffer)
    float32_bytes = len(torch_buffer.getvalue())
    weighted_error = sum(error * number for error, number in codec_errors) / max(count, 1)
    return {"bits_per_value": bits, "bytes_model": code_bytes + metadata_bytes, "parameter_count": count, "live_parameter_bytes": master, "diagnostic_code_payload_bytes": code_bytes, "codec_metadata_bytes": metadata_bytes, "checkpoint_bytes": checkpoint_bytes, "torch_checkpoint_bytes": float32_bytes, "encoding_error": weighted_error, "reconstruction_error": weighted_error, "encode_overhead_ms": encode_seconds * 1000, "decode_overhead_ms": decode_seconds * 1000, "storage_scope": "diagnostic per-tensor quantized payload plus codec metadata; checkpoint is lossless RGBD FP32"}, str(path)


def train_one(task, representation, seed, dataset, config, artifact_dir):
    device = config["device"]
    tx, ty, vx, vy, provenance = dataset
    dimensions = (tx.shape[-1], provenance["vocabulary_size"] if task == "language" else ty.shape[-1] if task in ("scalar", "periodic") else 10)
    model = model_for(task, representation, dimensions, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002 if task != "language" else 0.003)
    epochs = config["scalar_epochs"] if task in ("scalar", "periodic") else config["epochs"]
    batch_size = 64 if task == "language" else 256
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    scaler = torch.amp.GradScaler("cuda", enabled=representation == "fp16" and device.startswith("cuda"))
    curves = []
    initial = evaluate(model, vx, vy, task, representation, device)
    curves.append({"epoch": 0, "train_loss": None, "val_loss": initial["validation_loss"], "metric": initial.get("accuracy", initial.get("angular_mae_radians", initial["validation_loss"]))})
    train_seconds = 0.0
    trained_examples = 0
    observed = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) or module.__class__.__name__ in ("ChromaLinear", "ChromaEmbedding", "CodebookBottleneck"):
            def capture(module, args, output, key=name):
                if not module.training and isinstance(output, torch.Tensor):
                    observed[key] = output.detach().flatten()[:256].float().cpu()
            handles.append(module.register_forward_hook(capture))
            if len(handles) >= 4:
                break
    with PeakMemory(device) as memory:
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(len(tx), generator=generator, device=device)
            total, amount = 0.0, 0
            sync(device)
            start = time.perf_counter()
            for offset in range(0, len(tx), batch_size):
                indices = permutation[offset:offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(representation, device):
                    output = model(tx[indices])
                    loss = task_loss(task, output, ty[indices]) + auxiliary_loss(model)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss in {task}/{representation}/{seed}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(indices)
                amount += len(indices)
            sync(device)
            train_seconds += time.perf_counter() - start
            trained_examples += amount
            measured = evaluate(model, vx, vy, task, representation, device)
            curves.append({"epoch": epoch, "train_loss": total / amount, "val_loss": measured["validation_loss"], "metric": measured.get("accuracy", measured.get("angular_mae_radians", measured["validation_loss"]))})
    for handle in handles:
        handle.remove()
    model.eval()
    with torch.no_grad():
        def infer(current=model):
            with autocast_context(representation, device):
                return current(vx[:batch_size])
        _, inference_seconds = timed(infer, device, repeats=10)
    run_id = f"{task}-{representation}-s{seed}"
    storage, path = storage_metrics(model, representation, artifact_dir, run_id)
    metrics = {**measured, **storage, **memory.report(), "training_loss": curves[-1]["train_loss"], "initial_validation_loss": initial["validation_loss"], "training_seconds": train_seconds, "inference_batch_ms": inference_seconds * 1000, "training_throughput": trained_examples / train_seconds, "inference_throughput": min(batch_size, len(vx)) / inference_seconds, "epochs": epochs, "learning_rate": optimizer.param_groups[0]["lr"]}
    execution = module_quantization_report(model)
    execution["mixed_precision"] = representation if representation in ("fp16", "bf16") else None
    execution["optimizer_and_master_parameters"] = "FP32 Adam"
    execution["quantization_baseline"] = "INT8/INT4 are quantization-aware floating point training; separate integer inference record below" if representation in ("int8", "int4") else None
    record = {"id": run_id, "task": task, "representation": representation, "seed": seed, "metrics": metrics, "curves": curves, "execution": execution, "dataset": provenance, "artifact_path": path, "device": device,
              "model_config": {"schema_version": 1, "task": task, "representation": representation,
                               "seed": seed, "dimensions": list(dimensions)}}
    sample_parameter = next(model.parameters())
    samples = [tensor_sample(run_id + " weights", sample_parameter, experiment_id=run_id)]
    if sample_parameter.grad is not None:
        samples.append(tensor_sample(run_id + " gradients", sample_parameter.grad, experiment_id=run_id))
    for name, value in list(observed.items())[:2]:
        samples.append(tensor_sample(run_id + " activations " + name, value, experiment_id=run_id))
    actual_model = model.model if hasattr(model, "model") else model
    if hasattr(actual_model, "last_attention") and actual_model.last_attention is not None:
        samples.append(tensor_sample(run_id + " attention", actual_model.last_attention, experiment_id=run_id))
    for module in model.modules():
        if isinstance(module, nn.Embedding) or module.__class__.__name__ == "ChromaEmbedding":
            samples.append(tensor_sample(run_id + " embeddings", module.weight, experiment_id=run_id))
            break
    extra = None
    if representation == "fp32" and task in ("scalar", "mnist", "cifar"):
        integer = convert_integer_cpu(model)
        ix, iy = vx.cpu(), vy.cpu()
        measured_integer = evaluate(integer, ix, iy, task, "fp32", "cpu")
        with torch.no_grad():
            _, integer_seconds = timed(lambda: integer(ix[:256]), repeats=5)
        extra = {"id": run_id + "-integer-inference", "task": task, "seed": seed, "representation": "int8_integer_post_training", "device": "cpu", "dataset": provenance, "metrics": {**measured_integer, "inference_batch_ms": integer_seconds * 1000, "inference_throughput": min(256, len(ix)) / integer_seconds, "bytes_model": sum(p.numel() * p.element_size() for p in integer.state_dict().values()), "bits_per_value": 8}, "execution": {"linear_products": "INT32 matrix multiply from INT8 payloads", "scaling_bias_activations": "FP32", "input_encoding": "FP32 round/scale to INT32", "training": "post-training conversion of FP32 control", "scope": "CPU reference implementation, unoptimized"}, "curves": []}
        extra["source_record_id"] = run_id
    from .checkpoint_probe import write_checkpoint_probe
    write_checkpoint_probe(record, vx)
    del model, optimizer
    gc.collect()
    return record, [sample for sample in samples if sample is not None], extra


def periodic_data(seed, representation, device, count=4096):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.rand(count + 1024, generator=generator) * (2 * torch.pi)
    next_phase = phase + 0.4 + 0.3 * torch.sin(phase)
    target = torch.stack([next_phase.cos(), next_phase.sin()], -1)
    if representation == "scalar8":
        x = UniformCodec(bits=8, minimum=0, maximum=2 * math.pi).decode(UniformCodec(bits=8, minimum=0, maximum=2 * math.pi).encode(phase)).unsqueeze(-1) / math.pi - 1
        bits = 8
    elif representation == "raw_sincos64":
        x = torch.stack([phase.cos(), phase.sin()], -1)
        bits = 64
    else:
        # The numeric control uses exactly the same circular levels as hue.
        decoded = HueCodec(period=2 * math.pi).decode(HueCodec(period=2 * math.pi).encode(phase))
        if representation == "numeric_circular8":
            ids = torch.remainder(torch.round(phase.double() / (2 * math.pi) * 256).long(), 256)
            # Match the hue decoder's FP64 phase before the shared sin/cos cast.
            # Casting only this control to FP32 here breaks exact paired inputs.
            decoded = ids.double() / 256 * (2 * math.pi)
        x = torch.stack([decoded.cos(), decoded.sin()], -1).float()
        bits = 8
    from .common import dataset_hash
    info = {"source": "analytic phase rotation sequence", "formula": "theta_next=theta+0.4+0.3*sin(theta)", "train_examples": count, "validation_examples": 1024, "representation_bits": bits, "sha256": dataset_hash(phase, target), "input_transform": representation, "note": "sin/cos feature expansion and neural arithmetic use FP32"}
    return x[:count].float().to(device), target[:count].to(device), x[count:].float().to(device), target[count:].to(device), info
