from __future__ import annotations

import torch
from torch import nn
from ..tensor import ChromaTensor
from ..codecs import UniformCodec
from ..codecs.base import id_dtype


def _decoded(a, b, operation, codec=None):
    if not isinstance(a, ChromaTensor) or not isinstance(b, ChromaTensor):
        raise TypeError("operators require ChromaTensor operands")
    result = operation(a.decode(), b.decode())
    return ChromaTensor.from_tensor(result, codec or a.codec)


def chroma_add(a, b, codec=None):
    return _decoded(a, b, torch.add, codec)


def chroma_sub(a, b, codec=None):
    return _decoded(a, b, torch.sub, codec)


def chroma_mul(a, b, codec=None):
    return _decoded(a, b, torch.mul, codec)


def chroma_dot(a, b, codec=None):
    return _decoded(a, b, lambda x, y: torch.dot(x.flatten(), y.flatten()), codec)


def chroma_matmul(a, b, codec=None):
    return _decoded(a, b, torch.matmul, codec)


def chroma_similarity(a, b):
    """Decoded cosine similarity. This operation executes floating point."""
    return torch.nn.functional.cosine_similarity(a.decode().reshape(1, -1), b.decode().reshape(1, -1))[0]


def chroma_distance(a, b):
    """Decoded Euclidean distance. This operation executes floating point."""
    return torch.linalg.vector_norm(a.decode() - b.decode())


def chroma_activation(a, activation="relu", codec=None):
    functions = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "gelu": torch.nn.functional.gelu}
    if activation not in functions:
        raise ValueError("unknown activation")
    return ChromaTensor.from_tensor(functions[activation](a.decode()), codec or a.codec)


class LookupOperator:
    """A table over finite vocabulary IDs. Construction decodes; calls do not."""

    execution = {"construction": "floating point decode and quantize", "call": "integer indexing", "per_value_float_decode": False}

    def __init__(self, codec, operation="add", output_codec=None):
        if not isinstance(codec, UniformCodec) or codec.bits > 10:
            raise ValueError("lookup requires a uniform vocabulary of at most 1024 symbols")
        self.codec = codec
        self.output_codec = output_codec or codec
        functions = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}
        if operation not in functions:
            raise ValueError("unknown binary operation")
        self.operation = operation
        ids = torch.arange(2 ** codec.bits).reshape(-1, 1)
        values = codec.decode(ids)
        self.table = self.output_codec.encode(functions[operation](values[:, None], values[None, :]))

    def __call__(self, a, b):
        if a.codec.to_config() != self.codec.to_config() or b.codec.to_config() != self.codec.to_config():
            raise ValueError("lookup operands must use the table vocabulary")
        ids_a, ids_b = torch.broadcast_tensors(a.codes[..., 0].long(), b.codes[..., 0].long())
        table = self.table.to(a.device)
        if table.dtype == torch.uint16:
            codes = table.to(torch.int32)[ids_a, ids_b].to(torch.uint16)
        else:
            codes = table[ids_a, ids_b]
        return ChromaTensor(codes, self.output_codec, dtype=a.decoded_dtype)


def direct_grid_add(a, b):
    """Add same-scale nonnegative uniform codes with integer saturation."""
    codec = a.codec
    if not isinstance(codec, UniformCodec) or codec.minimum != 0 or codec.to_config() != b.codec.to_config():
        raise ValueError("direct grid addition requires equal uniform codecs with minimum zero")
    codes = (a.codes.long() + b.codes.long()).clamp(0, 2 ** codec.bits - 1).to(id_dtype(codec.levels))
    return ChromaTensor(codes, codec, dtype=a.decoded_dtype)


class LearnedOperator(nn.Module):
    """Learn a finite code mapping. Runtime uses floating point embeddings/MLP."""

    execution = {"call": "integer ID embedding lookup, FP32 MLP, argmax", "per_value_float_decode": False}

    def __init__(self, vocabulary_size=16, hidden=64):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.network = nn.Sequential(nn.Linear(32, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, vocabulary_size))

    def logits(self, a, b):
        return self.network(torch.cat([self.embedding(a.long()), self.embedding(b.long())], dim=-1))

    def forward(self, a, b):
        return self.logits(a, b).argmax(-1)

    def fit(self, lookup, steps=400, seed=0, learning_rate=0.02):
        if 2 ** lookup.codec.bits != self.vocabulary_size or lookup.output_codec.channels != 1:
            raise ValueError("learned operator and table vocabularies must match")
        torch.manual_seed(seed)
        device = next(self.parameters()).device
        ids = torch.arange(self.vocabulary_size, device=device)
        a, b = torch.meshgrid(ids, ids, indexing="ij")
        target = lookup.table[..., 0].long().to(device).flatten()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(self.logits(a.flatten(), b.flatten()), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses
