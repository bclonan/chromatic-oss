"""Color corruption, token corruption, sparsity, and equal-budget probes."""
from __future__ import annotations

import torch
from chromatensor import RGBCodec, UniformCodec, LearnedCodec, HierarchicalCodec
from chromatensor.autograd import ChromaQuantizer, rgb_to_codes, codes_to_rgb
from .common import seed_all, error_metrics


def run_robustness(config):
    seed_all(config["seeds"][0])
    values = torch.randn(65536).mul(0.25).clamp(-1, 1)
    rows = []
    for representation in ("int8", "chroma8", "fixed24", "rgb24"):
        state = ChromaQuantizer(representation=representation).encode(values)
        decoded = state.decode()
        rows.append({"representation": representation, "probe": "uncorrupted", **error_metrics(values, decoded), "code_zero_fraction": float((state.codes == 0).float().mean()), "source_zero_fraction": float((values == 0).float().mean()), "decoded_zero_fraction": float((decoded == 0).float().mean()), "nominal_bits": state.bits, "runtime_payload_bits": state.payload_bytes * 8 / len(values)})
        # Perturb the ID by one level. This checks the numerical grid, independent
        # of the byte palette used to label its values.
        maximum = (1 << state.bits) - 1
        perturbed = (state.codes.long() + 1).clamp(0, maximum)
        numerical = (perturbed.double() * state.scale + state.offset).float()
        rows.append({"representation": representation, "probe": "adjacent token ID", **error_metrics(decoded, numerical), "note": "same numerical-level perturbation for each matched precision pair"})
        # Flip the same least significant bit in the external payload. The
        # permutation used for chroma8 may turn a byte change into a distant ID.
        if state.rgb is not None:
            changed = state.rgb.clone()
            changed[..., -1] ^= 1
            ids = rgb_to_codes(changed, palette=representation != "rgb24")
            invalid = ids > maximum
            # Out-of-vocabulary RGB symbols are invalid. Report rejection rather
            # than silently pretending a palette color is a valid neighbor.
            valid = ~invalid
            reconstructed = (ids[valid].double() * state.scale + state.offset).float()
            error = error_metrics(decoded[valid], reconstructed) if valid.any() else {"mae": None, "mse": None, "max_error": None, "relative_mae": None}
            rows.append({"representation": representation, "probe": "payload LSB flip", **error, "invalid_symbol_fraction": float(invalid.float().mean()), "evaluated_valid_symbols": int(valid.sum())})
        else:
            changed = state.codes.long() ^ 1
            reconstructed = (changed.double() * state.scale + state.offset).float()
            rows.append({"representation": representation, "probe": "payload LSB flip", **error_metrics(decoded, reconstructed), "invalid_symbol_fraction": 0.0, "evaluated_valid_symbols": len(values)})
    relu = values.relu()
    for size in (16, 256):
        codec = LearnedCodec(size=size).fit(relu, iterations=8)
        decoded = codec.decode(codec.encode(relu))
        rows.append({"representation": f"learned_relu_{size}", "probe": "sparsity", **error_metrics(relu, decoded), "source_zero_fraction": float((relu == 0).float().mean()), "decoded_zero_fraction": float((decoded == 0).float().mean()), "used_vocabulary_fraction": float(codec.encode(relu).unique().numel() / size)})
    for tolerance in (0.1, 0.01, 0.001, 0.00001):
        codec = HierarchicalCodec(levels=4, tolerance=tolerance)
        codes = codec.encode(values)
        decoded = codec.decode(codes)
        rows.append({"representation": "adaptive_residual", "probe": "precision depth", "tolerance": tolerance, **error_metrics(values, decoded), "tokens_per_value": float(codes[..., -1].double().mean()), "dense_bytes_per_value": codes.shape[-1], "active_tokens_and_length_bytes_per_value": float(codes[..., -1].double().mean()) + 1})
    return rows
