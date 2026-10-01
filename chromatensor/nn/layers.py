"""Trainable PyTorch layers with discrete chromatic forward representations.

All dense kernels and normalization run in the input floating-point dtype.
These layers study fake quantization, not an integer-only neural network.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..autograd import ChromaQuantizer, codes_to_rgb


class _ExecutionMixin:
    def _record_execution(self, x: Tensor):
        self.execution_calls = getattr(self, "execution_calls", 0) + 1
        self.last_execution_dtype = str(x.dtype)


class ChromaLinear(_ExecutionMixin, nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *,
                 bits: int = 8, representation: str = "chroma", quantize_weights: bool = True,
                 quantize_activations: bool = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.weight_quantizer = ChromaQuantizer(bits, representation, enabled=quantize_weights)
        self.activation_quantizer = ChromaQuantizer(bits, representation, enabled=quantize_activations, range_axis=-1)

    def forward(self, x: Tensor) -> Tensor:
        self._record_execution(x)
        return F.linear(self.activation_quantizer(x), self.weight_quantizer(self.weight), self.bias)

    def to_torch(self) -> nn.Linear:
        layer = nn.Linear(self.in_features, self.out_features, self.bias is not None,
                          device=self.weight.device, dtype=self.weight.dtype)
        _copy_parameters(self, layer)
        return layer


class ChromaEmbedding(_ExecutionMixin, nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, max_norm=None,
                 norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False,
                 *, bits: int = 8, representation: str = "chroma", quantize_weights: bool = True,
                 quantize_activations: bool = True, device=None, dtype=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, device=device, dtype=dtype)
        # max_norm changes selected rows during forward. A row-local weight
        # range prevents another token's renormalization from changing this row.
        self.weight_quantizer = ChromaQuantizer(bits, representation, enabled=quantize_weights,
                                               range_axis=-1 if max_norm is not None else None)
        self.activation_quantizer = ChromaQuantizer(bits, representation, enabled=quantize_activations, range_axis=-1)

    def forward(self, x: Tensor) -> Tensor:
        self._record_execution(self.weight)
        # max_norm must update the master parameter as nn.Embedding does. Its
        # in-place renormalization on a fake-quantized output would violate autograd.
        if self.max_norm is not None:
            with torch.no_grad():
                torch.embedding_renorm_(self.weight, x, self.max_norm, self.norm_type)
        result = F.embedding(x, self.weight_quantizer(self.weight), self.padding_idx, None,
                             self.norm_type, self.scale_grad_by_freq, self.sparse)
        return self.activation_quantizer(result)

    def to_torch(self) -> nn.Embedding:
        layer = nn.Embedding(self.num_embeddings, self.embedding_dim, self.padding_idx,
                             self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse,
                             device=self.weight.device, dtype=self.weight.dtype)
        _copy_parameters(self, layer)
        return layer


class ChromaActivation(_ExecutionMixin, nn.Module):
    def __init__(self, activation: str | nn.Module = "relu", *, bits: int = 8,
                 representation: str = "chroma", quantize_activations: bool = True):
        super().__init__()
        if isinstance(activation, str):
            choices = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
                       "sigmoid": nn.Sigmoid, "silu": nn.SiLU, "identity": nn.Identity}
            if activation.lower() not in choices:
                raise ValueError(f"Unknown activation {activation!r}")
            activation = choices[activation.lower()]()
        if not isinstance(activation, nn.Module):
            raise TypeError("activation must be a supported name or torch.nn.Module")
        self.activation = activation
        self.activation_quantizer = ChromaQuantizer(bits, representation, enabled=quantize_activations, range_axis=-1)

    def forward(self, x: Tensor) -> Tensor:
        self._record_execution(x)
        return self.activation_quantizer(self.activation(x))

    def to_torch(self) -> nn.Module:
        return copy.deepcopy(self.activation)


class ChromaLayerNorm(_ExecutionMixin, nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, *, bits: int = 8, representation: str = "chroma",
                 quantize_weights: bool = True, quantize_activations: bool = True, device=None, dtype=None):
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device=device, dtype=dtype)
        self.weight_quantizer = ChromaQuantizer(bits, representation, enabled=quantize_weights)
        self.activation_quantizer = ChromaQuantizer(bits, representation, enabled=quantize_activations, range_axis=-1)

    def forward(self, x: Tensor) -> Tensor:
        self._record_execution(x)
        weight = self.weight_quantizer(self.weight) if self.weight is not None else None
        return F.layer_norm(self.activation_quantizer(x), self.normalized_shape, weight, self.bias, self.eps)

    def to_torch(self) -> nn.LayerNorm:
        parameter = self.weight if self.weight is not None else self.bias
        kwargs = {} if parameter is None else {"device": parameter.device, "dtype": parameter.dtype}
        layer = nn.LayerNorm(self.normalized_shape, self.eps, self.elementwise_affine,
                             self.bias is not None, **kwargs)
        _copy_parameters(self, layer)
        return layer


class CodebookBottleneck(_ExecutionMixin, nn.Module):
    """Learned vector quantization with an RGB symbol for each discrete code.

    Forward selects a code, then gathers its learned vector. Downstream dense
    layers still consume floats. Add auxiliary_loss() to the task loss to train
    both the codebook and the commitment objective.
    """

    def __init__(self, features: int, num_codes: int = 64, commitment_weight: float = 0.25,
                 *, device=None, dtype=None):
        super().__init__()
        if features < 1 or not 2 <= num_codes <= 65536:
            raise ValueError("features must be positive and num_codes must be between 2 and 65536")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be nonnegative")
        self.features, self.num_codes = features, num_codes
        self.commitment_weight = commitment_weight
        self.codebook = nn.Parameter(torch.empty(num_codes, features, device=device, dtype=dtype))
        nn.init.normal_(self.codebook, std=1 / math.sqrt(features))
        self.register_buffer("colors", codes_to_rgb(torch.arange(num_codes, device=device)))
        self.last_indices: Tensor | None = None
        self.last_rgb: Tensor | None = None
        self.last_auxiliary_loss: Tensor | None = None

    def nearest_codes(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.features:
            raise ValueError(f"Expected final dimension {self.features}, got {x.shape[-1]}")
        flat = x.reshape(-1, self.features)
        distances = flat.square().sum(1, keepdim=True) + self.codebook.square().sum(1) - 2 * flat @ self.codebook.t()
        return distances.argmin(1).reshape(x.shape[:-1])

    def forward(self, x: Tensor) -> Tensor:
        self._record_execution(x)
        indices = self.nearest_codes(x)
        quantized = F.embedding(indices, self.codebook)
        self.last_indices = indices.detach()
        self.last_rgb = self.colors[indices].detach()
        self.last_auxiliary_loss = F.mse_loss(quantized, x.detach()) + self.commitment_weight * F.mse_loss(x, quantized.detach())
        return x + (quantized - x).detach()

    def auxiliary_loss(self) -> Tensor:
        if self.last_auxiliary_loss is None:
            return self.codebook.sum() * 0
        return self.last_auxiliary_loss

    def usage(self) -> dict:
        if self.last_indices is None:
            return {"used_codes": 0, "num_codes": self.num_codes, "perplexity": 0.0}
        counts = torch.bincount(self.last_indices.reshape(-1), minlength=self.num_codes).float()
        probabilities = counts / counts.sum().clamp_min(1)
        nonzero = probabilities[probabilities > 0]
        return {"used_codes": int((counts > 0).sum()), "num_codes": self.num_codes,
                "perplexity": float(torch.exp(-(nonzero * nonzero.log()).sum()))}

    def __getstate__(self):
        state = super().__getstate__()
        # Cached training losses hold a graph and cannot be copied as parameters.
        state["last_auxiliary_loss"] = None
        return state


class ChromaMLP(_ExecutionMixin, nn.Module):
    def __init__(self, in_features: int, hidden_features: int | Sequence[int], out_features: int,
                 *, bits: int = 8, representation: str = "chroma", activation: str = "tanh",
                 quantize_weights: bool = True, quantize_activations: bool = True,
                 latent_codes: int | None = None, latent_weight: float = 0.25):
        super().__init__()
        hidden = [hidden_features, hidden_features] if isinstance(hidden_features, int) else list(hidden_features)
        if not hidden or any(width < 1 for width in hidden):
            raise ValueError("Provide at least one positive hidden width")
        dimensions = [in_features, *hidden, out_features]
        modules = []
        for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
            modules.append(ChromaLinear(source, target, bits=bits, representation=representation,
                                       quantize_weights=quantize_weights, quantize_activations=quantize_activations))
            if index < len(hidden):
                modules.append(ChromaActivation(activation, bits=bits, representation=representation,
                                                quantize_activations=quantize_activations))
                if latent_codes is not None:
                    modules.append(CodebookBottleneck(target, latent_codes, latent_weight))
        self.layers = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        self._record_execution(x)
        return self.layers(x)

    def auxiliary_loss(self) -> Tensor:
        terms = [module.auxiliary_loss() for module in self.modules() if isinstance(module, CodebookBottleneck)]
        return torch.stack(terms).sum() if terms else next(self.parameters()).sum() * 0

    def to_torch(self) -> nn.Sequential:
        # A learned VQ bottleneck has no exact nn.Linear equivalent; keep its
        # gather behavior, while replacing fake-quantized ordinary layers.
        return to_torch(self.layers)


def _copy_parameters(source: nn.Module, target: nn.Module):
    with torch.no_grad():
        for name, parameter in source.named_parameters(recurse=False):
            destination = getattr(target, name, None)
            if destination is not None:
                destination.copy_(parameter)
                destination.requires_grad_(parameter.requires_grad)
    target.train(source.training)


class ChromaModel(nn.Module):
    """Forward-compatible wrapper returned for a converted composite model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Keep any child-specific train/eval choices from the copied model.
        self.training = model.training

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to_torch(self) -> nn.Module:
        return to_torch(self.model)

    def auxiliary_loss(self) -> Tensor:
        terms = [m.auxiliary_loss() for m in self.modules() if isinstance(m, CodebookBottleneck)]
        return torch.stack(terms).sum() if terms else next(self.parameters()).sum() * 0


def convert_to_chroma(model: nn.Module, *, bits: int = 8, representation: str = "chroma",
                      quantize_weights: bool = True, quantize_activations: bool = True) -> nn.Module:
    """Copy a model and replace Linear, Embedding, and LayerNorm recursively.

    The original parameters and module are left untouched. Shared module references
    stay shared. Master parameter values and requires_grad flags are preserved.
    """
    copied = copy.deepcopy(model)
    memo: dict[int, nn.Module] = {}
    settings = dict(bits=bits, representation=representation, quantize_weights=quantize_weights,
                    quantize_activations=quantize_activations)

    def replace(module: nn.Module) -> nn.Module:
        if id(module) in memo:
            return memo[id(module)]
        if isinstance(module, (ChromaLinear, ChromaEmbedding, ChromaLayerNorm)):
            memo[id(module)] = module
            return module
        if isinstance(module, nn.MultiheadAttention):
            raise TypeError("Generic conversion cannot quantize nn.MultiheadAttention because its forward reads projection weights directly. Use ChromaAttention.from_torch and adapt its forward arguments explicitly.")
        for supported in (nn.Linear, nn.Embedding, nn.LayerNorm):
            if isinstance(module, supported) and type(module) is not supported:
                raise TypeError(f"Generic conversion cannot preserve custom {supported.__name__} subclass {type(module).__name__}; provide an explicit adapter")
        if isinstance(module, nn.Linear):
            result = ChromaLinear(module.in_features, module.out_features, module.bias is not None,
                                  device=module.weight.device, dtype=module.weight.dtype, **settings)
        elif isinstance(module, nn.Embedding):
            result = ChromaEmbedding(module.num_embeddings, module.embedding_dim, module.padding_idx,
                                     module.max_norm, module.norm_type, module.scale_grad_by_freq, module.sparse,
                                     device=module.weight.device, dtype=module.weight.dtype, **settings)
        elif isinstance(module, nn.LayerNorm):
            parameter = module.weight if module.weight is not None else module.bias
            kwargs = {} if parameter is None else dict(device=parameter.device, dtype=parameter.dtype)
            result = ChromaLayerNorm(module.normalized_shape, module.eps, module.elementwise_affine,
                                     module.bias is not None, **kwargs, **settings)
        else:
            result = module
            memo[id(module)] = result
            for name, child in list(module._modules.items()):
                if child is not None:
                    module._modules[name] = replace(child)
            return result
        _copy_parameters(module, result)
        # deepcopy already isolates the source and preserves every parameter
        # alias, including parameters registered on unchanged parent modules.
        for name, parameter in module._parameters.items():
            result._parameters[name] = parameter
        memo[id(module)] = result
        return result

    result = replace(copied)
    return result if hasattr(result, "to_torch") else ChromaModel(result)


def to_torch(model: nn.Module) -> nn.Module:
    """Copy a converted model back to ordinary floating-point master weights.

    Removing fake quantization can change outputs. This does not dequantize a
    packed checkpoint because trainable layers keep floating-point parameters.
    """
    if isinstance(model, ChromaModel):
        model = model.model
    copied = copy.deepcopy(model)
    memo: dict[int, nn.Module] = {}

    def replace(module: nn.Module) -> nn.Module:
        if id(module) in memo:
            return memo[id(module)]
        if isinstance(module, ChromaActivation):
            # Reuse the already isolated activation tree. Copying it again would
            # break ties to parameters registered elsewhere on the parent model.
            result = replace(module.activation)
        elif isinstance(module, (ChromaLinear, ChromaEmbedding, ChromaLayerNorm)):
            result = module.to_torch()
            for name, parameter in module._parameters.items():
                result._parameters[name] = parameter
        else:
            result = module
            memo[id(module)] = result
            for name, child in list(module._modules.items()):
                if child is not None:
                    module._modules[name] = replace(child)
        memo[id(module)] = result
        return result

    return replace(copied)
