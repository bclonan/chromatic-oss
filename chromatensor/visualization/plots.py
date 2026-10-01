from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tensor_triptych(original, chroma, path, title="Tensor representation"):
    x = original.detach().float().cpu().numpy().flatten()
    y = chroma.decode().detach().float().cpu().numpy().flatten()
    rgb = chroma.rgb.cpu().numpy().reshape(-1, 3)
    width = math.ceil(math.sqrt(len(x)))
    height = math.ceil(len(x) / width)
    padding = height * width - len(x)
    numerical = np.pad(x, (0, padding), constant_values=np.nan).reshape(height, width)
    colors = np.pad(rgb, ((0, padding), (0, 0))).reshape(height, width, 3)
    errors = np.pad(np.abs(x - y), (0, padding), constant_values=np.nan).reshape(height, width)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(numerical, cmap="coolwarm")
    axes[1].imshow(colors)
    axes[2].imshow(errors, cmap="magma")
    for axis, label in zip(axes, ("Original values", "Stored chromatic codes", "Absolute reconstruction error")):
        axis.set_title(label)
        axis.set_axis_off()
    figure.suptitle(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def training_curves(experiments, path, task="mnist"):
    figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for record in experiments:
        if record["task"] != task or not record.get("curves"):
            continue
        axes.plot([c["epoch"] for c in record["curves"]], [c["val_loss"] for c in record["curves"]], label=f"{record['representation']} s{record['seed']}", linewidth=1)
    axes.set(xlabel="Epoch", ylabel="Validation loss", title=task + " measured validation curves")
    if axes.lines:
        axes.legend(fontsize=7, ncols=2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def learned_manifold(book, path):
    values = np.asarray(book["values"])
    rgb = np.asarray(book["rgb"]) / 255
    frequencies = np.asarray(book["frequencies"])
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
    axes[0].scatter(np.arange(len(values)), values, c=rgb, s=12)
    axes[0].set(xlabel="Color token", ylabel="Learned scalar centroid", title=book["name"])
    axes[1].bar(np.arange(len(values)), frequencies, color=rgb, width=1)
    axes[1].set(xlabel="Color token", ylabel="Observed fit-sample frequency")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def result_overview(records, path):
    tasks = ["mnist", "cifar", "language", "scalar"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for task, axis in zip(tasks, axes.flatten()):
        rows = [r for r in records if r["task"] == task and r["representation"] in ("fp32", "int8", "chroma8", "rgb24", "fixed24")]
        representations = sorted(set(r["representation"] for r in rows))
        key = "accuracy" if task in ("mnist", "cifar") else "perplexity" if task == "language" else "validation_loss"
        means = [np.mean([r["metrics"][key] for r in rows if r["representation"] == representation]) for representation in representations]
        deviations = [np.std([r["metrics"][key] for r in rows if r["representation"] == representation]) for representation in representations]
        axis.bar(representations, means, yerr=deviations, color=["#79a8dd" if "chroma" not in label else "#d894ab" for label in representations])
        axis.set(title=task, ylabel=key)
    figure.savefig(path, dpi=160)
    plt.close(figure)
