"""Trainable chromatic experiments built on ordinary PyTorch kernels."""

from .attention import ChromaAttention
from .layers import ChromaActivation, ChromaEmbedding, ChromaLayerNorm, ChromaLinear, ChromaMLP, ChromaModel, CodebookBottleneck, convert_to_chroma, to_torch
from .report import execution_report, module_quantization_report

__all__ = ["ChromaLinear", "ChromaEmbedding", "ChromaActivation", "ChromaLayerNorm", "ChromaMLP", "ChromaAttention", "CodebookBottleneck", "ChromaModel", "convert_to_chroma", "to_torch", "module_quantization_report", "execution_report"]
