# What the measurements count

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The primary native hue library uses owned four-byte RGBD tokens and Rust integer kernels. Its current million-value proof compares equal 32-bit payload budgets, reports actual file bytes, and measures output allocation and integer control dtype promotion. It excludes construction, validation and persistence from operator timing. All measured native operators are slower than their NumPy controls. [The native guide](NATIVE_HUE.md) links current receipts. The remaining sections explain the legacy scalar codecs and PyTorch experiments.

ChromaTensor's training experiments keep floating-point master parameters. The
chromatic quantizer creates discrete RGB bytes in its forward pass, decodes them,
then calls ordinary PyTorch layers. Adam updates floating-point parameters and
optimizer states. Neither the RGB bytes nor an 8-bit vocabulary make this an
integer-only training system.

The optional neural RGBD mode has a different byte contract. It stores one actual
uint8 `[R, G, B, D]` row per encoded scalar and reconstructs finite FP32 values
exactly. The payload is four bytes per value. D is an exponent, and the state
needs no min/max range metadata. Its `codes` and `rgbd` fields reference the same
tensor, so the retained-byte report counts that buffer once. FP64 rounds to FP32.
The quantizer exposes `last_state.rgbd`, `payload_bytes`, `retained_state_bytes`
and `range_metadata_bytes`; the module report identifies `RGBD uint8 quadruples`.
Master parameters, decoded tensors, optimizer state and floating-point kernels
remain separate costs. This mode does not reduce the FP32 payload bit budget.

Default tokenizer RGBD rows also use four payload bytes per value. Their
`storage_nbytes` excludes serialized metadata and container framing. The current
million-token proof stores 4,000,000 payload bytes in a 4,000,295-byte file,
or 32.00236 total bits per value. Its independent oracle and cold-process decode
are storage correctness evidence, not neural latency or compression evidence.
[RGBD tokens](RGBD_TOKENS.md) gives the layout and proof receipt.

A chroma8 scalar has 256 uniform levels. Its current forward symbol is an RGB
byte triplet, so that symbol occupies 24 physical bits. The quantizer also keeps
the duplicate integer ID and range metadata for inspection. Neural activation
ranges cover each token's final feature vector, with FP64 offsets and scales.
Constant and singleton vectors retain their values through the offset. The result
JSON reports nominal levels, RGB payload bytes, retained inspection bytes, live master
parameter bytes, diagnostic scalar payload bytes, and checkpoint bytes separately.
An ideal packed 8-bit representation is a hypothetical size until code uses that
layout. The four-bit baseline also keeps one integer byte per code in its current
forward state. It does not pack two codes into a byte.

The diagnostic parameter payload encodes each parameter tensor after training.
It measures scalar reconstruction and encoded storage. It is separate from a
frozen executable model. The saved `.ctcheckpoint` files in these FP32-master
experiments preserve trained FP32 values in lossless RGBD records, plus integer
buffers and ZIP metadata. Default checkpoint export preserves FP64 through numeric
NPY members. An explicitly supplied codec instead chooses its own precision rules.
Those file sizes do not measure a deployed 8-bit neural model. The CPU integer
inference control is different. It has INT8 weight buffers, INT32 matrix products,
and FP32 scales, biases, activations, and input quantization.

The learned RGB encoder/decoder uses a numeric NPZ parameter checkpoint and a
JSON codec sidecar. Its 24-bit scalar payload is separate from those model files.
Instrumentation validates archive sizes, array dimensions, and numeric dtypes,
then loads arrays with pickle disabled. Its matrix estimate covers both neural
encoding and decoding on the full validation batch used by the timing record.

`forward_flops_estimate` counts two operations for each multiply/add term in the
model's dense matrix products on one inference batch. It includes QK and
probability-value matrix products for the tiny Transformer. Checkpoint weight
shapes determine the estimate when a valid checkpoint exists. Otherwise the
declared reference architecture determines it, and the record says so. Integer
inference products appear in `integer_multiply_add_estimate`, not floating FLOPs.

For codebook attention, inference replaces per-token QK products with learned
code-table reads. The estimate includes floating-point Q/K code assignment, the
rebuilt code-vector similarity table, and probability-value products. During
training, an extra floating-point QK calculation supplies surrogate gradients.
Nearest-code norms, argmin, table reads, softmax, RGB packing, rounding, range
estimation, normalization, bias, activations, memory traffic, and backward work
are outside these matrix-product counts. Causal masks do not reduce the dense
matrix products in this implementation.

Two checkpoint-backed CPU profiler forwards provide a second check, one scalar
FP32 model and one scalar chroma8 model. The report lists selected `aten`
operation calls, input shapes, and the FLOPs that PyTorch annotates. Those
annotations cover selected operations. They are neither exact instruction counts
nor a complete arithmetic total. The synthetic profiler inputs do not replace
the recorded validation dataset. Profiler time includes measurement overhead and
does not replace the separately timed inference batches.

Peak CPU memory means process RSS sampled every 10 milliseconds. It includes the
Python process and libraries. A CUDA record also reports PyTorch's peak allocated
device memory. Neither measure isolates one tensor or one representation.
Repeated runs can reuse allocator memory, so differences between small models
need more care than their raw byte totals suggest.

The paired controls matter more than the colors. Chroma8 and equal-level INT8 use
the same scalar quantization levels. RGB24 and fixed24 also share their scalar
levels. Hue8 and the numeric circular control now use identical input values.
Any difference between those pairs needs repeatable evidence before it can count
as an advantage. Faster integer kernels, smaller packed files, or better color
geometry would require their own implementation and measurements.

The trained codebook-attention experiment shows that a model can select learned
discrete RGB code states and learn a causal phase task. The current two-seed pilot
ran those models on CUDA. Its codebook mode had higher held-out error and longer
inference time than both dot-product controls at each seed. Mean batch time was
14.28647 ms for codebook attention, 8.94265 ms for chroma8 dot attention and
1.52259 ms for FP32 attention. Timing varies between runs, so retain the record's
hardware, batch size and execution precision when comparing it.
The two scalar profiler records are separate CPU executions. These records do not
establish a general speed or accuracy improvement for codebook attention.
