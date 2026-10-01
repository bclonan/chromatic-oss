"""Conventional attention and an experimental learned-code similarity path."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import ChromaLinear, CodebookBottleneck, _ExecutionMixin


class ChromaAttention(_ExecutionMixin, nn.Module):
    """Attention with chromatically fake-quantized projections.

    dot_product is conventional attention with optional fake quantization.
    codebook_similarity selects discrete Q/K codes and reads a table of learned
    code-vector dot products. Table construction, softmax, and value aggregation
    still use floats. The table is differentiable in training and rebuilt whenever
    used, so it does not silently become stale as the codebook changes.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, *, dropout: float = 0., bias: bool = True,
                 bits: int = 8, representation: str = "chroma", quantize_weights: bool = True,
                 quantize_activations: bool = True, similarity_mode: str = "dot_product",
                 num_codes: int = 32, commitment_weight: float = 0.25, batch_first: bool = True):
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        aliases = {"conventional": "dot_product", "codebook": "codebook_similarity"}
        similarity_mode = aliases.get(similarity_mode, similarity_mode)
        if similarity_mode not in {"dot_product", "codebook_similarity"}:
            raise ValueError("similarity_mode must be 'dot_product' or 'codebook_similarity'")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first, self.dropout, self.similarity_mode = batch_first, dropout, similarity_mode
        settings = dict(bits=bits, representation=representation, quantize_weights=quantize_weights,
                        quantize_activations=quantize_activations)
        self.q_proj = ChromaLinear(embed_dim, embed_dim, bias, **settings)
        self.k_proj = ChromaLinear(embed_dim, embed_dim, bias, **settings)
        self.v_proj = ChromaLinear(embed_dim, embed_dim, bias, **settings)
        self.out_proj = ChromaLinear(embed_dim, embed_dim, bias, **settings)
        self.codebook = CodebookBottleneck(self.head_dim, num_codes, commitment_weight) if similarity_mode == "codebook_similarity" else None
        self.last_attention: Tensor | None = None
        self.last_q_codes: Tensor | None = None
        self.last_k_codes: Tensor | None = None
        self.last_auxiliary_loss: Tensor | None = None

    def _split(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor | None = None, value: Tensor | None = None, *,
                attn_mask: Tensor | None = None, key_padding_mask: Tensor | None = None,
                need_weights: bool = False, is_causal: bool = False):
        self._record_execution(query)
        key = query if key is None else key
        value = key if value is None else value
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Attention inputs must have rank 3")
        if not self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        if key.shape[:2] != value.shape[:2] or query.shape[0] != key.shape[0]:
            raise ValueError("Attention batch dimensions and key/value sequence lengths must match")
        q, k, v = self._split(self.q_proj(query)), self._split(self.k_proj(key)), self._split(self.v_proj(value))
        scale = math.sqrt(self.head_dim)
        if self.codebook is None:
            scores = q @ k.transpose(-2, -1) / scale
        else:
            q_quantized = self.codebook(q)
            q_codes, q_loss = self.codebook.last_indices, self.codebook.auxiliary_loss()
            k_quantized = self.codebook(k)
            k_codes, k_loss = self.codebook.last_indices, self.codebook.auxiliary_loss()
            self.last_q_codes, self.last_k_codes = q_codes, k_codes
            self.last_auxiliary_loss = q_loss + k_loss
            table = self.codebook.codebook @ self.codebook.codebook.t() / scale
            lookup_scores = table[q_codes.unsqueeze(-1), k_codes.unsqueeze(-2)]
            # Forward is precisely the code-table lookup. A surrogate dot product
            # gives projections useful gradients through the discrete assignment.
            if torch.is_grad_enabled():
                surrogate_scores = q_quantized @ k_quantized.transpose(-2, -1) / scale
                # The lookup trains the code vectors themselves. The zero-valued
                # surrogate term supplies Q/K gradients through code assignment.
                scores = lookup_scores + (surrogate_scores - surrogate_scores.detach())
                self.last_score_execution = "code-table forward with floating-point QK gradient surrogate"
            else:
                scores = lookup_scores
                self.last_score_execution = "code-table scores without per-token QK dot product"
        batch, heads, queries, keys = scores.shape
        if is_causal:
            causal = torch.ones(queries, keys, device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            mask = attn_mask.to(scores.device)
            if mask.ndim == 3 and mask.shape[0] == batch * heads:
                mask = mask.reshape(batch, heads, queries, keys)
            elif mask.ndim == 3 and mask.shape[0] == batch:
                mask = mask.unsqueeze(1)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            elif mask.is_floating_point():
                scores = scores + mask
            else:
                raise TypeError("Attention masks must be boolean or floating point")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, keys):
                raise ValueError("key_padding_mask must have shape [batch, key_length]")
            mask = key_padding_mask.to(scores.device)[:, None, None, :]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            elif mask.is_floating_point():
                scores = scores + mask
            else:
                raise TypeError("Padding masks must be boolean or floating point")
        # Entirely masked rows produce zero, with finite gradients.
        fully_masked = torch.isneginf(scores).all(-1, keepdim=True)
        safe_scores = scores.masked_fill(fully_masked, 0.)
        attention = torch.softmax(safe_scores, -1).masked_fill(fully_masked, 0.)
        self.last_attention = attention.detach()
        attention = F.dropout(attention, self.dropout, self.training)
        result = (attention @ v).transpose(1, 2).reshape(batch, queries, self.embed_dim)
        result = self.out_proj(result)
        if not self.batch_first:
            result = result.transpose(0, 1)
        return (result, attention.mean(1)) if need_weights else result

    def auxiliary_loss(self) -> Tensor:
        return self.last_auxiliary_loss if self.last_auxiliary_loss is not None else self.q_proj.weight.sum() * 0

    @classmethod
    def from_torch(cls, source: nn.MultiheadAttention, **kwargs) -> "ChromaAttention":
        if source.in_proj_weight is None or source.bias_k is not None or source.bias_v is not None or source.add_zero_attn:
            raise ValueError("Conversion supports equal Q/K/V dimensions without extra key/value tokens")
        result = cls(source.embed_dim, source.num_heads, dropout=source.dropout,
                     bias=source.in_proj_bias is not None, batch_first=source.batch_first, **kwargs)
        result = result.to(device=source.in_proj_weight.device, dtype=source.in_proj_weight.dtype)
        with torch.no_grad():
            for index, projection in enumerate((result.q_proj, result.k_proj, result.v_proj)):
                projection.weight.copy_(source.in_proj_weight[index * source.embed_dim:(index + 1) * source.embed_dim])
                projection.weight.requires_grad_(source.in_proj_weight.requires_grad)
                if source.in_proj_bias is not None:
                    projection.bias.copy_(source.in_proj_bias[index * source.embed_dim:(index + 1) * source.embed_dim])
                    projection.bias.requires_grad_(source.in_proj_bias.requires_grad)
            result.out_proj.weight.copy_(source.out_proj.weight)
            result.out_proj.weight.requires_grad_(source.out_proj.weight.requires_grad)
            if source.out_proj.bias is not None:
                result.out_proj.bias.copy_(source.out_proj.bias)
                result.out_proj.bias.requires_grad_(source.out_proj.bias.requires_grad)
        result.train(source.training)
        return result

    def to_torch(self) -> nn.MultiheadAttention:
        if self.codebook is not None:
            raise ValueError("Codebook similarity has no exact ordinary MultiheadAttention equivalent")
        projections = (self.q_proj, self.k_proj, self.v_proj)
        if len({projection.weight.requires_grad for projection in projections}) != 1:
            raise ValueError("Packed torch Q/K/V weights cannot preserve different per-projection requires_grad flags")
        if self.q_proj.bias is not None and len({projection.bias.requires_grad for projection in projections}) != 1:
            raise ValueError("Packed torch Q/K/V biases cannot preserve different per-projection requires_grad flags")
        result = nn.MultiheadAttention(self.embed_dim, self.num_heads, self.dropout,
                                       self.q_proj.bias is not None, batch_first=self.batch_first,
                                       device=self.q_proj.weight.device, dtype=self.q_proj.weight.dtype)
        with torch.no_grad():
            result.in_proj_weight.copy_(torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)))
            if result.in_proj_bias is not None:
                result.in_proj_bias.copy_(torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)))
            result.out_proj.weight.copy_(self.out_proj.weight)
            if result.out_proj.bias is not None:
                result.out_proj.bias.copy_(self.out_proj.bias)
        result.in_proj_weight.requires_grad_(self.q_proj.weight.requires_grad)
        if result.in_proj_bias is not None:
            result.in_proj_bias.requires_grad_(self.q_proj.bias.requires_grad)
        result.out_proj.weight.requires_grad_(self.out_proj.weight.requires_grad)
        if result.out_proj.bias is not None:
            result.out_proj.bias.requires_grad_(self.out_proj.bias.requires_grad)
        result.train(self.training)
        return result

    def __getstate__(self):
        state = super().__getstate__()
        state["last_auxiliary_loss"] = None
        return state
