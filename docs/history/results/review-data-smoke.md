Historical result from the original research checkout. See docs/RELEASE_AUDIT.md for current verification.

# Measured results

Run `20261001T001904Z` used NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu126, profile `smoke`.

The record contains 11 model experiments and 0 numerical representation/distribution comparisons.

Every value below comes from the saved experiment records. Accuracy is a fraction. Training time excludes validation and export; GPU timings synchronize the device. Storage is a diagnostic quantized export estimate unless a row explicitly names an actual checkpoint or integer inference model. Inference-only records have no training time.

| Task | Representation | Seed | Validation loss | Task metric | Training seconds | Inference samples/s |
| --- | --- | --- | --- | --- | --- | --- |
| mnist | fp32 | 7 | 2.26216 | 0.359375 | 0.059 | 394246.5 |
| mnist | int8_integer_post_training | 7 | 2.26206 | 0.359375 | n/a | 14241.5 |
| mnist | int8 | 7 | 2.26227 | 0.359375 | 0.013 | 26230.7 |
| mnist | chroma8 | 7 | 2.26227 | 0.359375 | 0.018 | 16200.4 |
| cifar | fp32 | 7 | 2.36277 | 0.101562 | 0.007 | 143399.7 |
| cifar | int8_integer_post_training | 7 | 2.36319 | 0.101562 | n/a | 5698.5 |
| cifar | int8 | 7 | 2.36381 | 0.101562 | 0.025 | 15645.1 |
| cifar | chroma8 | 7 | 2.36381 | 0.101562 | 0.037 | 7070.7 |
| language | fp32 | 7 | 4.18199 | 65.4961 | 0.017 | 50058.3 |
| language | int8 | 7 | 4.18231 | 65.5172 | 0.042 | 4553.5 |
| language | chroma8 | 7 | 4.18231 | 65.5172 | 0.068 | 2378.2 |

## Paired controls

| Task | Comparison | Mean metric difference | Seeds |
| --- | --- | --- | --- |
| cifar | chroma8 minus int8 | 0 | 1 |
| cifar | chroma8 minus fp32 | 0 | 1 |
| language | chroma8 minus int8 | 0 | 1 |
| language | chroma8 minus fp32 | 0.0211780141 | 1 |
| mnist | chroma8 minus int8 | 0 | 1 |
| mnist | chroma8 minus fp32 | 0 | 1 |

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

- cifar: chroma8 minus int8 mean paired accuracy difference 0 across 1 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- language: chroma8 minus int8 mean paired perplexity difference 0 across 1 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- mnist: chroma8 minus int8 mean paired accuracy difference 0 across 1 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- Quantization-aware networks use floating point master parameters, optimizer state, matrix products, normalization, and softmax. The operator lookup and grid-add paths perform integer operations after their tables or scale metadata have been prepared.
- The executed profile is a seeded research pilot. Limited training, small models, and two seeds do not establish state-of-the-art accuracy, novel mathematics, end-to-end integer training, or a general speed advantage.
