# Limitations

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](docs/RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The primary `Tensor` API now calls the Rust native hue library. It stores actual RGBD tokens and computes in the finite ring `Z/391680Z`. [Native hue documentation](docs/NATIVE_HUE.md) defines this contract. The older `ChromaTensor` and neural APIs remain scalar quantization and codebook research controls.

## Native hue limits

The native library has 391,680 discrete hue states. Its operations differ from real-number arithmetic. Its modular multiplication has zero divisors because the modulus is composite. Hue blend follows the shortest circular arc and is not an associative ring operation.

Rust computes with integers on the CPU. This is not optical computation. No floating tensor, cached original scalar, Torch kernel or NumPy kernel supplies its operators. Exporting phases converts the stored colors to integer positions. The representation and library are implemented here; the underlying finite ring is established mathematics, and these checks do not establish research novelty.

General RGB encoding discards brightness and saturation. Gray has no hue and raises an error. D stores fine hue position. An RGB preview omits D and cannot alone recover a token. This D contract differs from the exponent byte in legacy scalar RGBD.

Each native token occupies 32 physical bits, equal to its uint32 control. The million-value proof uses seed 7331 and three timing repeats. Every measured native operation was slower than its NumPy integer control. The timing includes output allocation and control dtype promotion, and excludes input construction, validation and persistence. No compression, speed, energy or learning advantage has been shown.

The native API has no GPU backend, autograd, ordinary division, real-valued nonlinearities or neural training system. It currently packages x86-64 Windows and Linux libraries for Python 3.11 through 3.14. Linux loading still depends on compatible glibc. Its bounded, checksummed file loader is not an exhaustive hostile-file security audit. The synthetic dictionary retrieval example is deterministic and does not show learned inference or generalization.

The remaining sections describe the legacy numerical controls. Their earlier experimental receipts retain their original source identities and do not certify the native revision.

## Representation limits

RGB24 and equal-range 24-bit uniform fixed-point quantization reconstruct the same values. RGB packing provides an explicit color representation and visual inspection, while its precision comes from its 24-bit scalar grid.

Deterministic RGBD is a different byte arrangement of finite FP32. It costs 32 payload bits per scalar and preserves no more information than FP32. Float64 inputs round to FP32 before storage. The encoder rejects NaN, infinity, FP32 overflow, and nonzero values that round to zero. Default model checkpoint export handles FP64 separately through lossless numeric NPY members. An explicitly supplied RGBD codec keeps the FP32 precision limit.

The default tokenizer now uses real four-byte RGBD rows. Its RGB preview omits the exponent and cannot alone recover the value. `1.0` and `2.0` both preview as black. The explicit palette-ID tokenizer remains available through `vocabulary_size=256`; its numerical quantization is a separate contract. The optional neural RGBD mode stores exact finite FP32 forward bytes while retaining floating-point masters and kernels. It has no low-bit payload or integer-compute advantage.

Bounded RGB, HSV, uniform and learned scalar encoders clip inputs outside their declared ranges. Low error on a fitted or clipped sample does not prove coverage of unseen tails. Logarithmic RGB requires positive values.

Learned codebooks reject endpoints that collapse or become nonfinite in their parameter dtype. Logarithmic codecs reject endpoints with indistinguishable FP64 logarithms. A two-symbol hierarchical vocabulary supports one stage and rejects deeper configurations. These are explicit supported-domain limits.

HSV uses only its hue byte for scalar information, even though it stores three bytes. Its endpoint colors meet visually but decode to different scalar values. Hue intentionally identifies values separated by whole periods, so it cannot preserve the number of completed cycles. Circular and ordinary scalar error measure different things.

The scalar learned palette is a numerical centroid table with deterministic RGB labels. Fitting optimizes scalar squared error. It does not learn a perceptual color metric or establish that RGB neighbors have a useful scientific meaning. `LearnedChromaDecoder` gathers learned values, so its name must not be interpreted as a neural reconstruction network.

Learned RGBD fits one coarse table and one global additive residual table. It does not learn observation-specific confidence, uncertainty or semantic categories. A small residual error on its fit samples does not prove a benefit on a different distribution.

The jointly trained neural RGB codec learns actual channel coordinates and a neural reconstruction function. It has no uniform-grid reconstruction bound. Its three byte channels can collide for distinct observations. A functioning autoencoder may still reconstruct less accurately than an 8-bit uniform quantizer, despite paying a 24-bit payload and neural model overhead. The neural palette-decoder control tests omitted IDs, but a small omitted-ID set does not establish broad extrapolation.

## Storage and compression

Four-bit uniform codes currently occupy byte containers. Twenty-four-bit `UniformCodec` IDs occupy four-byte containers, while RGB24 uses three actual bytes. Report actual payload bytes as well as nominal bits. An ideal packed size is an estimate unless the measured file format produces it.

Dense hierarchical tensors retain every token slot and one depth byte. Adaptive precision saves token payload only in the compact `TokenSequence`, which also pays for its length bytes and codec metadata. A requested tolerance may remain unmet after the maximum number of stages.

Large learned vocabularies include floating-point centroids and RGB tables. Those tables can dominate small tensors and JSON file headers. Palette and range metadata count when comparing serialized files. Entropy compression can reward a distribution's repetitions without proving a benefit from chromatic organization.

Fake-quantized training retains floating-point weights, gradients and optimizer states. Inspector states also retain code IDs and sometimes redundant RGB bytes. Per-token feature ranges store FP64 offsets and scales. Their metadata can outweigh the payload for small feature vectors. A lower nominal code width therefore does not imply lower training memory.

## Computation and learning

The fake-quantized trainable layers use floating point for dense products, embeddings, nonlinearities, normalization, softmax, optimizer updates and gradients. Straight-through gradients approximate the derivative of discrete assignment, rounding and clipping. These models are not integer-only or fully chromatic. A separate converted CPU reference executes INT8 weights and INT32 products, with FP32 scales, biases, and activations. Its execution and timing records are separate from fake-quantized training.

Activation ranges use each token's final feature vector, so future tokens and unrelated batch examples do not change its range. The model still needs the correct causal mask. Generic layer conversion preserves weight aliases but rejects ordinary multi-head attention and custom subclasses of replaced layers. An explicit attention adapter has a different forward argument and return contract, so callers must adapt their calls.

Lookup arithmetic avoids per-value numerical decoding at runtime, but constructing its table uses numerical decoding and floating-point arithmetic. Table storage grows with the square of vocabulary size. The implementation restricts lookup vocabularies to a practical size. Learned operators use floating-point embeddings and MLPs even though their inputs and outputs are token IDs.

Codebook attention still performs floating-point code assignment and table construction. It also keeps floating-point softmax and value aggregation. Its training gradient surrogate computes QK products. Any inference timing claim must include assignment and table costs, not only table indexing.

The tensor wrapper supports named shape operations and explicit PyTorch conversion. It is not a drop-in implementation of every `torch.Tensor` operation. Slicing gives ordinary PyTorch-style views where possible, so mutating shared code storage can change both views. Codecs and codebook parameters are mutable; changing a codebook changes existing tensors that reference it. Saved token sequences and files retain their own codec configuration.

The legacy reference implementation alone does not justify specialized-kernel performance claims. The separate native Rust hue kernels now have measured correctness and timing. Claims about fewer operations or faster inference still require measurements or explicit operation accounting.

## Evidence limits

The test suite checks implemented contracts, gradients, representation validity and selected CPU/CUDA behavior. It does not prove behavior for every shape, distribution, device or dtype. CUDA tests skip when no device is available. Seeded runs improve reproducibility but do not guarantee identical floating-point results across software and hardware versions.

Task results apply to the recorded dataset splits, sample budgets, seeds and training settings. A small pilot cannot establish full-dataset accuracy, production performance or superiority over tuned quantization methods. Same-seed same-data controls are necessary. Multiple seeds estimate variance within the tested setup; they do not establish general performance across models and datasets. The sustained release procedure uses four seeds and withholds its official MNIST test access until new training completes. Its acceptance requires a completed current-source record, and it discloses prior repository use of that test split.

A dashboard preview is evidence that the inspector works on its submitted values. Fitting a codebook on those same values demonstrates fitting; it does not measure held-out reconstruction. Installed-wheel CPU inference replay checks the saved model in its recorded environment. Production traffic, public hosting, other hardware, and human screen-reader usability require separate evidence.

## File integrity

All four formats use JSON and numeric buffers without pickle. SHA-256 digests and ZIP CRC checks detect corruption and provide no sender authentication. Callers can lower file, metadata, payload, and archive expansion budgets. [File formats and limits](docs/FILE_FORMATS.md) records the exact defaults and dtype policy. These budgets do not cap parser CPU time, learned parameter memory, validation scratch space, peak RAM, or device copies. The file tests cover forged metadata and bounded expansion; they do not establish exhaustive fuzz coverage or an audited hostile-file processing service.

Model checkpoints contain state dictionaries. Reloading still requires a compatible model architecture. Returning a fake-quantized model to ordinary PyTorch removes forward quantization, which can change its outputs. A vector codebook cannot in general be replaced by an ordinary linear layer without changing its behavior.

