# Representation theory

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](docs/RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

ChromaTensor represents scalar values with finite symbols and explicit decoding rules. The codecs study quantization, periodic geometry, residual precision and learned tables. Ordinary hardware bits and arithmetic still implement every path.

## Scalar and tensor maps

Let C be a finite set of chromatic symbols. An encoder E maps a supported real domain X into C^d, and a decoder D maps C^d back into real values. Reconstruction error is |x - D[E[x]]|. Square brackets in this document denote function application.

For a tensor with shape n1 by ... by nk, encoding acts independently on its scalar entries. Its stored shape is n1 by ... by nk by d. The last dimension contains code channels and is outside the logical tensor shape.

Every codec rejects NaN and infinity. Bounded codecs clip finite values outside their declared domain. RGBD rejects values that overflow or underflow its FP32 domain. Clipping error and quantization error must therefore be reported separately when inputs exceed a bounded domain.

## RGB24

Set Q = 2^24 - 1 and choose finite a < b. For x inside the domain, define t = [x - a] / [b - a] and q = round[Q t]. The encoder stores

```text
R = floor[q / 65536]
G = floor[q / 256] mod 256
B = q mod 256
```

The decoder first recovers q = 65536 R + 256 G + B, then returns a + [b - a] q / Q.

The real-arithmetic absolute error bound is [b - a] / [2Q]. The implementation performs normalization and reconstruction in FP64. Returning FP32, FP16 or BF16 introduces that dtype's rounding as a separate effect. Values outside the domain first clip to an endpoint and have no in-domain error guarantee relative to the original value.

RGB24 is numerically identical to 24-bit uniform fixed-point quantization with the same endpoints. The RGB arrangement does not improve its scalar reconstruction accuracy. Comparing it only with FP32 would overlook both 8-bit alternatives and this equal-bit control.

For a logarithmic positive domain, replace x, a and b with their logarithms during normalization. Reconstruction exponentiates the result. Within the domain, relative error is at most exp[log[b/a] / [2Q]] - 1 in real arithmetic. Nonpositive inputs are invalid.

The implementation also rejects logarithmic endpoints whose FP64 logarithms coincide. A finite pair of distinct endpoints alone does not guarantee a usable logarithmic grid in machine arithmetic.

## Deterministic RGBD

`RGBDCodec` preserves each finite FP32 bit pattern, including negative zero and subnormal numbers. D stores the IEEE-754 biased exponent e. The three RGB bytes store a sign bit s and 23 fraction bits f. Normal values have 24 significand bits because their leading one is implicit.

```text
RGB packed integer = [s << 23] | f
D = e
Normal value = [-1]^s * 2^[e - 127] * [1 + f / 2^23]
Subnormal value = [-1]^s * 2^-126 * [f / 2^23]
```

D=0 covers zero and subnormal values. D=255 is invalid because it selects nonfinite IEEE values. D therefore has a numerical meaning rather than acting as an arbitrary extra byte. FP64 source values first round to FP32. Values too large to remain finite or too small to remain nonzero cause an error. The stored representation has 32 bits per scalar and no uncompressed payload saving relative to FP32.

## HSV and circular hue

`HSVCodec` stores a bounded scalar position in H, with S=V=255. Its payload occupies 24 bits but has only eight effective scalar bits. H=0 and H=255 decode to different endpoints, although both render as red on the hue circle. This codec tests the cost and visual ambiguity of that mapping.

`HueCodec` instead encodes a periodic phase. With period P, origin o and K bins, its ID is round[K * [x-o mod P] / P] mod K. Decode returns o + P q / K. Negative values and values beyond one period intentionally share symbols with their equivalent phases.

The appropriate error is the shortest circular distance, d_P[x,y] = |[x-y+P/2 mod P] - P/2|. Its real-arithmetic bound is P / [2K]. A large ordinary absolute error near the wrap point can coexist with a small circular error. Periodic experiments therefore report circular errors and compare sin/cos outputs rather than using scalar MAE alone.

## Learned scalar and RGBD tables

`LearnedChromaCodebook` has K trainable scalar centroids c_j and a unique RGB symbol for each ID j. Its numerical objective on fit observations x_i is mean_i min_j |x_i-c_j|^2. Fitting uses quantile initialization, nearest-centroid assignments and Lloyd mean updates. Sorted centroid boundaries permit binary-search assignment without an observations-by-K distance matrix. Large observation sets use a seeded subsample with a recorded maximum size.

The centroid dtype must retain distinct finite domain endpoints. Saved configurations retain parameter dtypes, including FP64, independently of the process default dtype. That preserves the table's precision; it does not remove assignment or reconstruction error.

RGB labels come from a deterministic Morton grid. They identify learned centroids; the fitting loss does not optimize RGB distance or a perceptual color objective. Adjacent scalar ranks often occupy nearby grid cells, but color proximity is not a guaranteed numerical metric.

`LearnedChromaEncoder` returns nearest-centroid integer IDs. `LearnedChromaDecoder` gathers trainable centroid values. This decoder is a learned table lookup. It is not a neural reconstruction network. Gradients can update selected centroid values, while the discrete encoder has no ordinary derivative.

`LearnedRGBDCodec` fits a coarse scalar table c_j and a residual table r_k. RGB is the coarse centroid's unique symbol. D selects the additive residual. The decoder returns clip[c_j+r_k,a,b]. The residual table fits x_i-c_j after fitting the coarse table. D does not represent confidence or a learned scientific property beyond that additive correction.

The separate vector bottleneck in `chromatensor.nn` assigns vectors to learned code vectors with squared Euclidean distance. Its task training adds codebook and commitment losses and uses a straight-through gradient for downstream computation. This follows the established vector-quantization approach described by [van den Oord, Vinyals and Kavukcuoglu](https://arxiv.org/abs/1711.00937).

## Jointly learned RGB representation

`LearnedRGBCodec` is a separate encoder-decoder experiment. A scalar MLP produces three sigmoid channels z in [0,1]. Its discrete symbol is round[255 z]. An RGB-to-scalar MLP reconstructs the value from those normalized bytes. Joint training minimizes numerical squared error plus a small squared difference between colors of neighboring scalar observations.

Byte rounding has an identity straight-through gradient during fitting. The actual inference encoder emits uint8 RGB, and the inference decoder consumes those RGB bytes. This learns the channel coordinates and reconstruction network together, rather than relabeling fitted centroids with fixed colors. Its numerical precision has no RGB24 uniform-grid error guarantee. Payload storage is 24 bits per scalar, plus the neural model's weights and configuration.

`LearnedNeuralDecoder` in `chromatensor.decoders` is an independent supervised RGB-to-scalar MLP. The decoder experiment trains it on a subset of palette IDs and measures omitted IDs separately. It tests whether fixed color labels admit useful neural reconstruction. It does not turn the scalar Lloyd table into a jointly learned RGB encoder.

Both neural paths train on numerical training observations and report independent validation observations. Their decoder kernels use floating point. Successful fitting establishes a functioning learned representation, while a comparison with equal-bit fixed-point and 8-bit uniform controls tests whether it is useful.

## Hierarchical residual precision

For K scalar symbols, stage zero quantizes [a,b]. Each following stage quantizes the preceding residual in a symmetric range. At level j>0, the residual radius is [b-a] / [2[K-1]^j]. After L active stages, the real-arithmetic error bound is [b-a] / [2[K-1]^L]. Floating-point roundoff can matter when very fine stages approach machine precision.

Residual refinement requires K greater than two. With K=2, the bound does not shrink with another stage, so the implementation accepts only one stage for that vocabulary.

A tolerance can stop refinement once measured reconstruction error is small enough. Reaching the maximum level can still leave error above tolerance. The code records the actual number of active stages for every value.

`HierarchicalCodec` stores L byte slots and one depth byte per scalar, including zero-filled unused slots. Its dense storage does not shrink when refinement stops. `TokenSequence` omits inactive slots and retains one length byte per value. Report actual token bytes plus length bytes, not only log2[K] times the number of active tokens. No bit packing or entropy coding is implied.

## Operators and gradients

Decode-compute-encode operators perform floating-point numerical work. A lookup operator precomputes a finite pair table using floating point, then its calls use integer ID indexing. Direct grid addition performs saturated integer code addition for equal nonnegative uniform grids. Integer-only validation keeps those result wrappers from silently decoding each output.

Learned operators use integer IDs followed by floating-point embeddings and an MLP. Codebook attention builds a floating-point code-vector similarity table, selects entries by integer Q/K codes, then runs floating-point softmax and value aggregation. Training also computes a floating-point surrogate QK product for gradients.

The fake-quantized layers round and reconstruct values in forward while approximating backward derivatives with an identity straight-through estimator. Parameters, gradients, optimizer state and dense kernels remain floating point. The distinction between simulated quantization during training and deployed integer computation is consistent with [PyTorch's FakeQuantize documentation](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html).

Neural activation quantizers derive their range within each token's final feature vector. They store the offset and scale in FP64. Constant and singleton vectors retain their values through the offset. Future tokens cannot change an earlier token's range, but attention still needs the correct causal mask. Range metadata and inspection buffers count toward actual memory even when a code has only eight nominal bits.

## Cube geometry and global encoding

The cube is a discrete tensor grid. A quarter turn in two equally sized axes permutes their coordinates inside one selected layer of another axis. The operation applies the same permutation to color codes and cell identities. Its inverse restores the prior state exactly. Four repeated quarter turns give the identity permutation. Odd turns require equal plane extents, while half turns also support rectangular planes.

The distributed representation applies an orthonormal Fourier transform across the numerical tensor's axes. Every frequency coefficient is a weighted sum over all positions. The saved field quantizes each coefficient's phase to 256 circular bins and its magnitude to 65,536 inclusive levels. Phase and magnitude need three byte channels plus one scale for the whole cube. A spatial impulse affects every coefficient, and each retained coefficient contributes across the inverse transform.

Erasing coefficients leaves an underdetermined inverse problem. Zero filling chooses zero for the missing coefficients. A trained decoder adds assumptions learned from its data. Neither operation establishes unique recovery of arbitrary missing information. The equal-bit numerical phase/magnitude control uses identical coefficients and gives identical reconstruction. [The implementation and its sources](docs/HOLOGRAPHIC.md) document full-spectrum redundancy and floating point execution.

## What would support the research hypothesis

Useful evidence would show a measurable benefit beyond equal-bit uniform quantization or ordinary vector quantization. Candidates include periodic reconstruction under circular distance, adaptive precision after including metadata, or useful computation through finite operator tables. A color relabeling with identical accuracy and additional overhead is a valid negative result. Tests establish implementation behavior; measured task and runtime results belong in `RESULTS.md`.

