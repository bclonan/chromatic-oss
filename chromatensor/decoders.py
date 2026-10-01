"""Supervised RGB neural reconstruction, separate from scalar centroid lookup."""
from __future__ import annotations

import math

import torch
from torch import nn

from .codecs.base import finite_values, validate_codes, validate_domain


class LearnedNeuralDecoder(nn.Module):
    """An RGB-to-scalar MLP whose weights train against numerical targets.

    Color labels alone need not carry an easy numerical geometry. This decoder
    can therefore fail to match exact codebook lookup, especially for unseen IDs.
    """

    def __init__(self, minimum: float = -1, maximum: float = 1, hidden: int = 64, seed: int = 0):
        super().__init__()
        self.minimum, self.maximum = validate_domain(minimum, maximum)
        fp32_bounds = torch.tensor([self.minimum, self.maximum], dtype=torch.float32)
        if (not torch.isfinite(fp32_bounds).all() or fp32_bounds[0] == fp32_bounds[1]
                or self.maximum - self.minimum > torch.finfo(torch.float32).max):
            raise ValueError("Neural decoder endpoints must be distinct finite FP32 values")
        if isinstance(hidden, bool) or int(hidden) != hidden or not 4 <= hidden <= 256:
            raise ValueError("hidden must be an integer in [4,256]")
        self.hidden = int(hidden)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                         nn.Tanh(), nn.Linear(hidden, 1), nn.Tanh())
        self.fit_history: list[dict] = []

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        validate_codes(rgb, 3)
        parameter = next(self.parameters())
        normalized = rgb.to(dtype=parameter.dtype, device=parameter.device) / 255 * 2 - 1
        values = self.network(normalized).squeeze(-1)
        return self.minimum + (values + 1) / 2 * (self.maximum - self.minimum)

    def fit(self, rgb: torch.Tensor, targets: torch.Tensor, steps: int = 300, seed: int = 0,
            learning_rate: float = 0.01, validation_rgb: torch.Tensor | None = None,
            validation_targets: torch.Tensor | None = None) -> "LearnedNeuralDecoder":
        if isinstance(steps, bool) or int(steps) != steps or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("steps and learning_rate must be positive")
        steps = int(steps)
        validate_codes(rgb, 3)
        parameter = next(self.parameters())
        targets = finite_values(targets).to(device=parameter.device, dtype=parameter.dtype)
        if rgb.shape[:-1] != targets.shape or targets.numel() == 0:
            raise ValueError("RGB and target shapes must match and contain observations")
        if (validation_rgb is None) != (validation_targets is None):
            raise ValueError("Supply both validation RGB and targets")
        if validation_targets is not None:
            validate_codes(validation_rgb, 3)
            validation_targets = finite_values(validation_targets).to(device=parameter.device, dtype=parameter.dtype)
            if validation_rgb.shape[:-1] != validation_targets.shape or validation_targets.numel() == 0:
                raise ValueError("Validation RGB and target shapes must match")
        # There is no dropout or randomized batching in this full-batch fit.
        # seed records the caller's reproducibility setting; initialization has
        # its own constructor seed to avoid resetting an already trained model.
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.fit_history = []
        self.train()
        for step in range(steps):
            optimizer.zero_grad()
            loss = (self(rgb) - targets).square().mean()
            loss.backward()
            optimizer.step()
            if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 20) == 0:
                with torch.no_grad():
                    train_loss = float((self(rgb) - targets).square().mean())
                    validation_loss = None if validation_rgb is None else float((self(validation_rgb) - validation_targets).square().mean())
                self.fit_history.append({"step": step + 1, "training_loss": train_loss,
                                         "validation_loss": validation_loss, "seed": int(seed)})
        self.eval()
        return self

