"""Jointly trained scalar-to-RGB encoder and RGB-to-scalar neural decoder."""
from __future__ import annotations

import math

import torch
from torch import nn

from .base import ScalarCodec, finite_values, validate_codes, validate_domain


_FLOAT_DTYPES = {str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}


class LearnedRGBCodec(nn.Module, ScalarCodec):
    """A learned three-channel byte representation with explicit FP32 decoding.

    The encoder and decoder learn together through byte quantization with an
    identity straight-through estimator. Its 24-bit payload is learned RGB,
    rather than a fixed color label attached to a scalar centroid. Runtime
    neural encoding and reconstruction still perform floating-point arithmetic.
    """

    name = "learned_rgb"
    channels = 3
    bits_per_value = 24

    def __init__(self, minimum: float = -1, maximum: float = 1, hidden: int = 32, seed: int = 0):
        super().__init__()
        self.minimum, self.maximum = validate_domain(minimum, maximum)
        fp32_bounds = torch.tensor([self.minimum, self.maximum], dtype=torch.float32)
        if (not torch.isfinite(fp32_bounds).all() or fp32_bounds[0] == fp32_bounds[1]
                or self.maximum - self.minimum > torch.finfo(torch.float32).max):
            raise ValueError("Neural RGB endpoints must be distinct finite FP32 values")
        if isinstance(hidden, bool) or int(hidden) != hidden or not 4 <= hidden <= 256:
            raise ValueError("hidden must be an integer in [4,256]")
        self.hidden = int(hidden)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.encoder = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                         nn.Tanh(), nn.Linear(hidden, 3), nn.Sigmoid())
            self.decoder = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                         nn.Tanh(), nn.Linear(hidden, 1), nn.Tanh())
        self.fit_history: list[dict] = []
        self.fit_metadata: dict = {}

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(self.minimum, self.maximum) - self.minimum) / (self.maximum - self.minimum) * 2 - 1

    def _denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.minimum + (normalized + 1) / 2 * (self.maximum - self.minimum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Neural chromatic training requires finite floating-point input")
        normalized = self._normalize(x).reshape(-1, 1)
        latent = self.encoder(normalized)
        quantized = torch.round(latent * 255) / 255
        straight_through = latent + (quantized - latent).detach()
        return self._denormalize(self.decoder(straight_through)).reshape(x.shape)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        values = finite_values(x)
        model_parameter = next(self.parameters())
        normalized = self._normalize(values).reshape(-1, 1).to(device=model_parameter.device, dtype=model_parameter.dtype)
        chunks = []
        for chunk in normalized.split(16384):
            latent = self.encoder(chunk)
            if not torch.isfinite(latent).all():
                raise ValueError("Neural RGB encoder produced nonfinite channel values")
            chunks.append(torch.round(latent * 255).clamp(0, 255).to(torch.uint8))
        result = torch.cat(chunks, 0) if chunks else torch.empty((0, 3), dtype=torch.uint8, device=model_parameter.device)
        return result.reshape(*values.shape, 3).to(values.device)

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        return validate_codes(codes, 3)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.validate(codes)
        model_parameter = next(self.parameters())
        normalized_rgb = codes.reshape(-1, 3).to(device=model_parameter.device, dtype=model_parameter.dtype) / 255
        chunks = [self._denormalize(self.decoder(chunk)) for chunk in normalized_rgb.split(16384)]
        result = torch.cat(chunks, 0) if chunks else torch.empty((0, 1), dtype=model_parameter.dtype, device=model_parameter.device)
        if not torch.isfinite(result).all():
            raise ValueError("Neural RGB decoder produced nonfinite values")
        return result.reshape(codes.shape[:-1]).to(codes.device)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        self.validate(codes)
        return codes.to(torch.uint8)

    def fit(self, samples: torch.Tensor, iterations: int = 300, seed: int = 0,
            batch_size: int = 256, learning_rate: float = 0.01, smoothness_weight: float = 0.001,
            validation_samples: torch.Tensor | None = None) -> "LearnedRGBCodec":
        if (isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1
                or isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or not math.isfinite(smoothness_weight) or smoothness_weight < 0):
            raise ValueError("Training steps, batch size and learning rate must be positive")
        iterations, batch_size = int(iterations), int(batch_size)
        parameter = next(self.parameters())
        samples = finite_values(samples).reshape(-1).clamp(self.minimum, self.maximum).to(device=parameter.device, dtype=parameter.dtype)
        if samples.numel() < 2:
            raise ValueError("Learned RGB fitting needs at least two observations")
        validation = None if validation_samples is None else finite_values(validation_samples).reshape(-1).to(device=parameter.device, dtype=parameter.dtype)
        if validation is not None and validation.numel() == 0:
            raise ValueError("Validation observations must be nonempty when supplied")
        generator = torch.Generator().manual_seed(int(seed))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.fit_history = []
        self.train()
        for step in range(iterations):
            indices = torch.randint(samples.numel(), (min(batch_size, samples.numel()),), generator=generator).to(samples.device)
            batch = samples[indices]
            optimizer.zero_grad()
            normalized = self._normalize(batch).reshape(-1, 1)
            latent = self.encoder(normalized)
            quantized = torch.round(latent * 255) / 255
            reconstructed = self._denormalize(self.decoder(latent + (quantized - latent).detach())).reshape(-1)
            reconstruction_loss = (reconstructed - batch).square().mean()
            ordered = latent[batch.argsort()]
            smoothness_loss = (ordered[1:] - ordered[:-1]).square().mean() if ordered.shape[0] > 1 else latent.sum() * 0
            loss = reconstruction_loss + smoothness_weight * smoothness_loss
            loss.backward()
            optimizer.step()
            if step == 0 or step == iterations - 1 or (step + 1) % max(1, iterations // 20) == 0:
                with torch.no_grad():
                    train_mse = (self(samples) - samples).square().mean()
                    validation_mse = None if validation is None else float((self(validation) - validation).square().mean())
                self.fit_history.append({"step": step + 1, "training_loss": float(train_mse),
                                         "validation_loss": validation_mse, "batch_loss": float(reconstruction_loss.detach()),
                                         "smoothness_loss": float(smoothness_loss.detach())})
        self.fit_metadata = {"steps": iterations, "seed": int(seed), "training_samples": samples.numel(),
                             "validation_samples": 0 if validation is None else validation.numel(),
                             "batch_size": batch_size, "learning_rate": learning_rate,
                             "smoothness_weight": smoothness_weight,
                             "gradient_method": "identity straight-through for byte rounding"}
        self.eval()
        return self

    def to_config(self) -> dict:
        return {"name": self.name, "minimum": self.minimum, "maximum": self.maximum, "hidden": self.hidden,
                "state": {key: value.detach().cpu().tolist() for key, value in self.state_dict().items()},
                "state_dtypes": {key: str(value.dtype) for key, value in self.state_dict().items()},
                "fit_history": self.fit_history, "fit_metadata": self.fit_metadata}

    @classmethod
    def from_config(cls, config: dict) -> "LearnedRGBCodec":
        result = cls(config["minimum"], config["maximum"], config["hidden"])
        if set(config["state"]) != set(result.state_dict()):
            raise ValueError("Neural RGB state has unexpected or missing parameters")
        dtypes = config.get("state_dtypes")
        if dtypes is not None and (not isinstance(dtypes, dict) or set(dtypes) != set(config["state"])):
            raise ValueError("Neural RGB parameter dtype metadata must match the saved state")
        state = {}
        for name, expected in result.state_dict().items():
            dtype = _FLOAT_DTYPES.get("torch.float32" if dtypes is None else dtypes[name])
            if dtype is None:
                raise ValueError("Unsupported neural RGB parameter dtype")
            value = torch.tensor(config["state"][name], dtype=dtype)
            if value.shape != expected.shape or not torch.isfinite(value).all():
                raise ValueError("Neural RGB state has invalid shape or nonfinite values")
            state[name] = value
        result.load_state_dict(state, assign=True)
        result.fit_history = list(config.get("fit_history", []))
        result.fit_metadata = dict(config.get("fit_metadata", {}))
        result.eval()
        return result

