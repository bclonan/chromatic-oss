from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import threading
import time

import numpy as np
import psutil
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment():
    from .provenance import git_identity, dependency_versions
    git = git_identity()
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "threads": torch.get_num_threads(), "revision": git["commit"], "git": git, "dependencies": dependency_versions(), "timestamp": datetime.now(timezone.utc).isoformat()}


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def timed(function, device="cpu", repeats=5, warmup=2):
    for _ in range(warmup):
        function()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = function()
    sync(device)
    return result, (time.perf_counter() - start) / repeats


class PeakMemory:
    """Sample process RSS every 10 ms and track allocator peak on CUDA."""
    def __init__(self, device):
        self.device = device
        self.stop_event = threading.Event()
        self.process = psutil.Process()
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss

    def _sample(self):
        while not self.stop_event.wait(0.01):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.cuda_peak = torch.cuda.max_memory_allocated(self.device) if str(self.device).startswith("cuda") else None

    def report(self):
        return {"peak_memory_bytes": self.peak_rss, "rss_increase_bytes": max(0, self.peak_rss - self.start_rss), "peak_cuda_allocated_bytes": self.cuda_peak, "memory_method": "process RSS sampled at 10 ms; CUDA allocator peak if available"}


def error_metrics(original, decoded, circular_period=None):
    original, decoded = original.detach().double(), decoded.detach().double()
    delta = decoded - original
    if circular_period:
        delta = torch.remainder(delta + circular_period / 2, circular_period) - circular_period / 2
    absolute = delta.abs()
    return {"mae": float(absolute.mean()), "mse": float(delta.square().mean()), "max_error": float(absolute.max()) if absolute.numel() else 0.0, "relative_mae": float((absolute / original.abs().clamp_min(1e-8)).mean())}


def dataset_hash(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    if (isinstance(value, dict) and value.get("schema_version") == 1
            and isinstance(value.get("config"), dict) and isinstance(value.get("experiments"), list)
            and value.get("status") in {"running", "complete", "failed"}):
        from .provenance import bind_result_evidence
        bind_result_evidence(value, output_path=path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    try:
        # Windows readers can briefly deny replacement. Keep the previous valid
        # snapshot available while retrying the atomic replacement.
        for attempt in range(8):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.02 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)


def autocast_context(representation, device):
    if representation not in ("fp16", "bf16"):
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.float16 if representation == "fp16" else torch.bfloat16)

