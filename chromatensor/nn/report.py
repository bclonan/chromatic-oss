"""Honest execution and storage accounting for chromatic neural experiments."""

from __future__ import annotations

from torch import nn

from ..autograd import ChromaQuantizer
from .layers import CodebookBottleneck


def module_quantization_report(model: nn.Module) -> dict:
    parameters = list(model.parameters())
    report = {
        "training_method": "fake quantization with identity straight-through gradients",
        "fully_chromatic": False,
        "floating_point_execution": True,
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        "parameter_count": sum(p.numel() for p in parameters),
        "master_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "quantizers": [],
        "codebooks": [],
        "attention": [],
        "executed_modules": [],
        "limitations": [
            "Dense, embedding, normalization, nonlinearities, softmax, and gradient kernels use floating point.",
            "Master parameters and optimizer states stay floating point; estimated code bytes are not measured training-memory savings.",
            "Chromatic uniform palettes relabel the same scalar levels as equal-bit uniform integer quantization.",
            "Straight-through gradients approximate discrete assignment, rounding, and clipping.",
        ],
    }
    for name, module in model.named_modules():
        if isinstance(module, ChromaQuantizer):
            state = module.last_state
            report["quantizers"].append({
                "module": name, "representation": module.representation, "bits": module.bits,
                "enabled": module.enabled, "forward_calls": module.forward_calls,
                "configured_range_axis": module.range_axis,
                "range_method": state.range_method if state is not None else "RGBD bit packing" if module.representation == "rgbd" else "fixed range" if module.min_value is not None else "global min/max" if module.range_axis is None else "axis min/max",
                "last_range_axes": list(state.range_axes) if state is not None else None,
                "last_range_metadata_bytes": state.range_metadata_bytes if state is not None else 0,
                "last_range_shape": list(state.offset.shape) if state is not None else None,
                "last_constant_ranges": state.constant_ranges if state is not None else 0,
                "encoded_values": module.encoded_values,
                "last_payload_bytes": state.payload_bytes if state is not None else 0,
                "last_retained_inspection_bytes": state.retained_state_bytes if state is not None else 0,
                "last_ideal_packed_code_bytes": (state.codes.numel() // 4 * module.bits + 7) // 8 if state is not None and module.representation == "rgbd" else (state.codes.numel() * module.bits + 7) // 8 if state is not None else 0,
                "payload_format": "RGBD uint8 quadruples" if module.representation == "rgbd" else "RGB uint8 triples" if module.representation in {"chroma", "chroma8", "rgb24"} else "integer indices",
            })
        if isinstance(module, CodebookBottleneck):
            report["codebooks"].append({"module": name, "features": module.features,
                                       "codebook_bytes": module.codebook.numel() * module.codebook.element_size(),
                                       "color_bytes": module.colors.numel(),
                                       "last_rgb_bytes": module.last_rgb.numel() if module.last_rgb is not None else 0,
                                       "last_index_bytes": module.last_indices.numel() * module.last_indices.element_size() if module.last_indices is not None else 0,
                                       "last_vector_count": module.last_indices.numel() if module.last_indices is not None else 0,
                                       **module.usage()})
        if hasattr(module, "similarity_mode"):
            report["attention"].append({"module": name, "similarity_mode": module.similarity_mode,
                                        "last_score_execution": getattr(module, "last_score_execution", "floating-point QK dot product"),
                                        "assignment_and_table_dtype": str(module.q_proj.weight.dtype),
                                        "softmax_and_value_aggregation": "floating point"})
        if hasattr(module, "execution_calls"):
            report["executed_modules"].append({"module": name, "class": type(module).__name__,
                                               "forward_calls": module.execution_calls,
                                               "kernel_dtype": module.last_execution_dtype})
    return report


execution_report = module_quantization_report
