from __future__ import annotations

import math
import torch
from torch import nn


class StandardMLP(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))

    def forward(self, x):
        return self.layers(x)


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary, hidden=48, length=32):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.position = nn.Embedding(length, hidden)
        # Explicit Q/K/V layers allow recursive conversion of every projection.
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocabulary)
        self.register_buffer("causal_mask", torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1))
        self.last_attention = None

    def forward(self, tokens):
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(x.shape[-1])
        scores = scores.masked_fill(self.causal_mask[:tokens.shape[1], :tokens.shape[1]], float("-inf"))
        attention = torch.softmax(scores, -1)
        self.last_attention = attention.detach()
        x = self.norm1(x + self.projection(attention @ v))
        x = self.norm2(x + self.ff(x))
        return self.head(x)


class IntegerLinear(nn.Module):
    """CPU INT8 payloads, INT32 products, then FP32 scaling and bias."""
    def __init__(self, linear):
        super().__init__()
        scale = linear.weight.detach().abs().amax(1).clamp_min(1e-8) / 127
        self.register_buffer("weight_codes", (linear.weight.detach() / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else torch.zeros(linear.out_features))

    def forward(self, x):
        scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
        codes = (x / scale).round().clamp(-127, 127).to(torch.int32)
        products = codes @ self.weight_codes.to(torch.int32).T
        return products.float() * scale * self.scale + self.bias


def convert_integer_cpu(model):
    import copy
    result = copy.deepcopy(model).cpu()
    def visit(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, IntegerLinear(child))
            else:
                visit(child)
    visit(result)
    return result
