Historical result from the original research checkout. See docs/RELEASE_AUDIT.md for current verification.

# Measured results

Run `bench-20261001T042347877523Z-b33dfab1880b` used NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu126, profile `smoke`.

The record contains 1 model experiments and 0 numerical representation/distribution comparisons.

Every value below comes from the saved experiment records. Accuracy is a fraction. Training time excludes validation and export; GPU timings synchronize the device. Storage is a diagnostic quantized export estimate unless a row explicitly names an actual checkpoint or integer inference model. Inference-only records have no training time.

| Task | Representation | Seed | Validation loss | Task metric | Training seconds | Inference samples/s |
| --- | --- | --- | --- | --- | --- | --- |
| scalar | chroma8 | 31 | 0.252018 | 0.252018 | 0.168 | 22958.6 |

## Paired controls

| Task | Comparison | Mean metric difference | Seeds |
| --- | --- | --- | --- |

## Precision on 4,096 bounded values per distribution

The suite records unclipped out-of-domain sample counts, but evaluates quantization on the same clipped values for each bounded codec. Hue uses circular error, so its MAE does not compare directly with linear error.

| Distribution | Codec | MAE | Max error | Actual code bits/value | Metadata bytes |
| --- | --- | --- | --- | --- | --- |

## Operator measurements

| Method | MAE | Batch latency ms | Values/s |
| --- | --- | --- | --- |

## Actual file storage

| Format | File bytes | Separate metadata bytes | Bits/value including metadata | Reconstruction MAE |
| --- | --- | --- | --- | --- |

## Instrumentation

The JSON includes per-model matrix operation estimates, live master bytes, runtime RGB buffers, diagnostic exports, and partial PyTorch CPU profiler observations from trained scalar checkpoints. The estimates exclude uncounted operations and are not hardware instruction counts. See docs/MEASUREMENT.md for the complete scope.

## Interpretation

- Quantization-aware networks use floating point master parameters, optimizer state, matrix products, normalization, and softmax. The operator lookup and grid-add paths perform integer operations after their tables or scale metadata have been prepared.
- The executed profile is a seeded research pilot. Limited training, small models, and two seeds do not establish state-of-the-art accuracy, novel mathematics, end-to-end integer training, or a general speed advantage.
