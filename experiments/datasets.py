from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request
import numpy as np
import torch
from .common import dataset_hash


def stratified_indices(labels, count, seed):
    labels = torch.as_tensor(labels)
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a nonempty one-dimensional array")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    generator = torch.Generator().manual_seed(seed)
    classes = labels.unique(sorted=True)
    count = min(count, len(labels))
    if count == len(labels):
        return torch.randperm(len(labels), generator=generator)
    if count == 0:
        return torch.empty(0, dtype=torch.int64)
    chosen = []
    remaining = []
    for index, label in enumerate(classes):
        candidates = torch.where(labels == label)[0]
        take = count // len(classes) + int(index < count % len(classes))
        permutation = candidates[torch.randperm(len(candidates), generator=generator)]
        selected = min(take, len(candidates))
        chosen.append(permutation[:selected])
        remaining.append(permutation[selected:])
    joined = torch.cat(chosen)
    shortfall = count - joined.numel()
    if shortfall:
        available = torch.cat(remaining)
        additions = available[torch.randperm(len(available), generator=generator)[:shortfall]]
        joined = torch.cat((joined, additions))
    return joined[torch.randperm(len(joined), generator=generator)]


def image_data(task, root, train_count, val_count, seed, device):
    from torchvision.datasets import CIFAR10, MNIST
    dataset_type = MNIST if task == "mnist" else CIFAR10
    train = dataset_type(root, train=True, download=True)
    validation = dataset_type(root, train=False, download=True)
    train_ids = stratified_indices(train.targets, train_count, seed)
    val_ids = stratified_indices(validation.targets, val_count, seed + 1)

    def extract(data, ids):
        source = torch.as_tensor(np.array(data.data))
        x = source[ids].float() / 255
        if task == "cifar":
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.unsqueeze(1)
        y = torch.as_tensor(data.targets)[ids].long()
        return x.flatten(1), y

    tx, ty = extract(train, train_ids)
    vx, vy = extract(validation, val_ids)
    info = {"source": "torchvision.datasets." + dataset_type.__name__, "source_url": "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html" if task == "mnist" else "https://www.cs.toronto.edu/~kriz/cifar.html", "partition": "stratified subset of official train and official test splits", "train_examples": len(tx), "validation_examples": len(vx), "selection_seed": seed, "sha256": dataset_hash(tx, ty, vx, vy), "preprocessing": "pixels / 255, flattened; no augmentation"}
    return tx.to(device), ty.to(device), vx.to(device), vy.to(device), info


def scalar_data(seed, device, count=4096):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(count + 1024, 2, generator=generator) * 2 - 1
    y = torch.stack([torch.sin(x[:, 0] * torch.pi), x[:, 0].square(), torch.sin(x[:, 0] * 3) * torch.cos(x[:, 1] * 2) + 0.2 * x[:, 1].square()], -1)
    info = {"source": "analytic functions", "targets": ["sin(pi*x0)", "x0^2", "sin(3*x0)*cos(2*x1)+0.2*x1^2"], "train_examples": count, "validation_examples": 1024, "sha256": dataset_hash(x, y)}
    return x[:count].to(device), y[:count].to(device), x[count:].to(device), y[count:].to(device), info


def language_data(seed, root, count, val_count, device, sequence_length=32):
    path = Path(root) / "tinyshakespeare.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    vocabulary = sorted(set(text))
    encode = {char: index for index, char in enumerate(vocabulary)}
    corpus = torch.tensor([encode[char] for char in text], dtype=torch.long)
    split = int(len(corpus) * 0.9)
    generator = torch.Generator().manual_seed(seed)

    def samples(stream, amount):
        starts = torch.randint(len(stream) - sequence_length - 1, (amount,), generator=generator)
        offsets = torch.arange(sequence_length)
        return stream[starts[:, None] + offsets], stream[starts[:, None] + offsets + 1]

    tx, ty = samples(corpus[:split], count)
    vx, vy = samples(corpus[split:], val_count)
    info = {"source": "Tiny Shakespeare, char-rnn repository", "source_url": url, "corpus_sha256": hashlib.sha256(raw).hexdigest(), "sha256": dataset_hash(tx, ty, vx, vy), "partition": "first 90% train, last 10% validation, sampled windows", "train_examples": count, "validation_examples": val_count, "sequence_length": sequence_length, "vocabulary_size": len(vocabulary), "vocabulary": "".join(vocabulary)}
    return tx.to(device), ty.to(device), vx.to(device), vy.to(device), info

