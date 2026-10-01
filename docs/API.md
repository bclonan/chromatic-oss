# Python API

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The primary native API is `Tensor`, also named `HueTensor` and `ColorTensor`. Use [the native hue guide](NATIVE_HUE.md) for its Rust encoder, decoder, RGBD tokens and dependency-free installation. The examples below document the retained legacy numerical APIs. Their D byte means an FP32 exponent, while native hue D means fine hue position.

Install the repository with `pip install -e .`. These examples use PyTorch tensors and the reference Python implementation.

## Integer-backed tensor

```python
import torch
from chromatensor import ChromaTensor, RGBCodec, RGBDCodec

x = torch.randn(32, 128)
cx = ChromaTensor.from_tensor(x)
assert torch.equal(cx.decode(), x)

print(cx.shape)          # torch.Size([32, 128])
print(cx.dtype)          # integer storage dtype
print(cx.decoded_dtype)  # original numerical output dtype
print(cx.device)
print(cx.storage_nbytes)
print(cx.codes.shape)    # [32, 128, 4] for default RGBD
print(cx.rgbd.shape)     # [32, 128, 4], the actual saved token bytes
print(cx.rgb.shape)      # [32, 128, 3]

bounded = ChromaTensor.from_tensor(x, RGBCodec(minimum=-4, maximum=4))
numeric = bounded.to_torch(dtype=torch.float32)
```

The default RGBD codec preserves finite FP32 bit patterns. `.rgbd` exposes all four actual code bytes and requires the finite FP32 RGBD codec. `.rgb` exposes a display preview. D stores the biased exponent, so the RGB preview alone cannot decode a value. FP64 inputs round to FP32; widening the decoded dtype does not restore lost precision. Explicit bounded codecs clip finite inputs outside their domains. Encoding is discrete and drops the input's autograd graph.

Shape operations keep code channels out of the value shape.

```python
from chromatensor.tensor import cat, stack

row = cx[0]
columns = cx[:, 8:16]
flat = cx.reshape(-1)
transposed = cx.transpose(0, 1)
joined = cat([cx, cx], dim=0)
batch = stack([cx, cx], dim=0)
copied = cx.clone()
moved = cx.to("cuda") if torch.cuda.is_available() else cx.cpu()
```

`ChromaTensor.cat` and `ChromaTensor.stack` are also available. Concatenation and stacking require equivalent codecs and matching numerical output dtypes. Use `decode(dtype=...)` for numerical dtype conversion. `to` moves integer storage and accepts an optional `decoded_dtype` keyword; it does not cast code bytes into floats.

Slicing supports Python and NumPy boolean scalars, boolean masks, ellipsis and new axes without indexing code channels. A scalar boolean adds an axis. Boolean lists and NumPy boolean arrays become boolean mask tensors. Use an explicit tensor mask for a multidimensional boolean list to make its meaning clear.

## Scalar codecs

```python
from chromatensor.codecs import (
    RGBCodec, RGBDCodec, UniformCodec, HSVCodec, HueCodec,
    LearnedCodec, LearnedRGBDCodec, LearnedRGBCodec, HierarchicalCodec, codec_from_config,
)

codecs = [
    RGBCodec(-1, 1),
    RGBCodec(1e-6, 1e6, logarithmic=True),
    RGBDCodec(),
    UniformCodec(bits=8, minimum=-1, maximum=1),
    HSVCodec(-1, 1),
    HueCodec(period=2 * torch.pi),
    LearnedCodec(size=256, minimum=-1, maximum=1),
    LearnedRGBDCodec(size=64, modifier_size=16, minimum=-1, maximum=1),
    LearnedRGBCodec(minimum=-1, maximum=1, hidden=32, seed=7),
    HierarchicalCodec(levels=3, vocabulary_size=16, minimum=-1, maximum=1,
                      tolerance=0.001),
]
```

Each codec follows the same contract.

| Member | Result |
| --- | --- |
| `encode(x)` | Integer tensor with `x.shape` plus the final `channels` dimension |
| `decode(codes)` | Numerical tensor with the final code dimension removed |
| `colors(codes)` | uint8 RGB tensor with numerical shape plus three channels |
| `validate(codes)` | Integer-only symbol and shape validation |
| `to_config()` | JSON-compatible codec metadata, including learned tables |
| `channels` | Stored code channels per scalar |
| `bits_per_value` | Nominal code budget; inspect container bytes separately |

`codec_from_config(codec.to_config())` recreates its representation. RGB24 and uniform decoding use FP64 calculations; RGBD reconstructs FP32. `ChromaTensor.decode` then casts to its recorded numerical output dtype.

Logarithmic RGB and uniform codecs require distinct FP64 logarithms for their positive endpoints. They reject a range that collapses after that transform. Learned configurations record their numerical parameter dtypes, so reconstruction does not depend on the current default PyTorch dtype. Legacy configurations without dtype fields restore FP32 parameters.

For hue, use `codec.circular_error(original, reconstructed)` instead of ordinary absolute error when the target is periodic. `HierarchicalCodec.token_lengths(codes)` returns the actual active depth for each value. Dense hierarchical storage still contains every configured slot.

Hierarchical codecs accept one stage with two symbols. Multiple stages require at least three symbols. Two-symbol residual stages do not shrink the error range and now raise `ValueError`. A requested tolerance can still remain unmet when the selected stage count provides too little precision.

## Learned scalar tables

```python
samples = torch.randn(10000).clamp(-1, 1)
codec = LearnedCodec(size=64).fit(samples, iterations=20, seed=7)
cx = ChromaTensor.from_tensor(samples, codec)
loss = (cx.decode() - samples).square().mean()
history = codec.codebook.fit_history

residual_codec = LearnedRGBDCodec(size=16, modifier_size=16).fit(samples, seed=7)
```

The fit method uses deterministic Lloyd updates and a seeded subsample for large fit sets. `maximum_samples` controls that fit budget. Fit on training observations and evaluate reconstruction on separate observations when making accuracy claims.

```python
from chromatensor.codecs import (
    LearnedChromaCodebook, LearnedChromaEncoder, LearnedChromaDecoder,
)

book = LearnedChromaCodebook(size=16)
encoder = LearnedChromaEncoder(book)
decoder = LearnedChromaDecoder(book)
ids = encoder(torch.tensor([-0.8, 0.1, 0.9]))
reconstruction = decoder(ids)
reconstruction.sum().backward()
assert book.values.grad is not None
```

The decoder is a trainable value lookup. Its selected centroids receive gradients. The encoder's integer assignment has no ordinary gradient. `book.palette` gives unique RGB symbols; `book.ids_from_colors` reverses those symbols into IDs.

`LearnedChromaCodebook` accepts `dtype=torch.float64` when its numerical domain needs FP64 centroids. Its default is the current PyTorch default dtype. Construction rejects endpoints that become nonfinite or indistinguishable in the chosen dtype. For example, an FP32 codebook rejects the domain `[1e-50, 2e-50]` rather than silently replacing every centroid with zero.

```python
book = LearnedChromaCodebook(size=16, minimum=1e-50, maximum=2e-50,
                            dtype=torch.float64)
codec = LearnedCodec(codebook=book)
```

## Neural RGB representation and reconstruction

```python
from chromatensor.codecs import LearnedRGBCodec
from chromatensor.decoders import LearnedNeuralDecoder

train = torch.rand(8192) * 2 - 1
validation = torch.rand(4096) * 2 - 1
neural_codec = LearnedRGBCodec(hidden=32, seed=7)
neural_codec.fit(train, iterations=300, seed=7, validation_samples=validation)
colors = neural_codec.encode(validation)
reconstructed = neural_codec.decode(colors)
history = neural_codec.fit_history
```

`LearnedRGBCodec` learns the three channel coordinates and its reconstruction MLP together. Its stored codes are uint8 RGB. Fitting uses a straight-through estimator for byte quantization. Encoding and decoding run floating-point MLPs; no fixed-point error bound applies. Move the codec module with `neural_codec.to(device)` to choose its neural execution device. Its output code and value tensors preserve the caller's tensor device.

The separate `LearnedNeuralDecoder` accepts byte RGB in `forward`. Train it with `fit(rgb, numerical_targets, steps=300, seed=7)` and optional paired `validation_rgb` and `validation_targets`. Its decoder network receives ordinary gradients. This class reconstructs supplied color symbols; it does not learn a numerical-to-color encoder.

`experiments.decoder.run_decoder_experiment(config, artifacts=...)` returns a measured record, tensor inspector samples and a learned manifold. It records held-out reconstruction, comparison controls, model bytes, timing and omitted-ID palette-decoder errors. Saved decoder NPZ files contain fixed-network weights, with JSON metadata beside them. A matching network can load these arrays with `numpy.load(..., allow_pickle=False)` and `load_state_dict`.

## Vocabulary and compact tokens

```python
from chromatensor import ChromaTokenizer, ChromaDecoder, TokenSequence

values = torch.tensor([1.0, 2.0], dtype=torch.float32)
sequence = ChromaTokenizer().encode(values)
assert sequence.tokens.tolist() == [[0, 0, 0, 127], [0, 0, 0, 128]]
assert sequence.tokens.dtype == torch.uint8
assert sequence.num_tokens == 2
assert sequence.storage_nbytes == 8
assert torch.equal(ChromaDecoder().decode(sequence), values)

sequence.save("artifacts/tokens.ctensor", compress=False)
loaded = TokenSequence.load("artifacts/tokens.ctensor", device="cpu",
                            max_payload_bytes=1024 * 1024)
assert torch.equal(ChromaDecoder().decode(loaded.rgbd), values)
```

`ChromaTokenizer()` now defaults to actual RGBD byte rows with shape `[N, 4]`. `sequence.rgbd` restores the original numerical shape plus four channels, and `.rgb` returns the first three display channels. `num_tokens` counts rows. No original floating tensor is retained. The convenience save/load pair uses the checked RGBD `.ctensor` format and the public loader budgets; it rejects palette/residual sequences and non-RGBD files. Raw `uint8` tensors with a trailing four-channel dimension also decode through `ChromaDecoder()`.

This changes the earlier no-argument tokenizer default. Use `ChromaTokenizer(vocabulary_size=256)` to keep bounded scalar IDs, or supply a vocabulary explicitly.

```python
from chromatensor.tokenizer import ChromaVocabulary

vocabulary = ChromaVocabulary(size=65536)
tokenizer = ChromaTokenizer(vocabulary)
sequence = tokenizer.encode(torch.linspace(-1, 1, 1000))
reconstructed = ChromaDecoder().decode(sequence)

symbol = vocabulary.symbol(42)
assert vocabulary.token_id(symbol) == 42
print(sequence.tokens, sequence.shape, sequence.storage_nbytes)
```

Explicit vocabulary sizes from 2 through 65,536 are supported. The requested 16, 64, 256, 1,024, 4,096 and 65,536 sizes all use the same path. `ChromaVocabulary.fit(samples, seed=...)` fits a learned vocabulary. `ChromaCodebook` is the scalar table class available from the tokenizer package. A symbol uses the actual palette bytes in hexadecimal `COLOR_RRGGBB`; its integer ID is a separate lookup index.

```python
adaptive = ChromaTokenizer(vocabulary_size=16, levels=3, tolerance=0.001)
sequence = adaptive.encode(torch.linspace(-1, 1, 1000))
print(sequence.tokens_per_value)
print(sequence.lengths)
print(sequence.storage_nbytes)
reconstructed = adaptive.decode(sequence)
```

Residual sequences store only active tokens plus per-value lengths. Their configured vocabulary must have at most 256 symbols. Each `TokenSequence` keeps its numerical shape, codec configuration and output dtype. `sequence.to(device)` moves its integer buffers.

Passing a vocabulary, vocabulary size, minimum, maximum, multiple levels or tolerance selects this explicit ID path. Default RGBD uses D as an exponent; `LearnedRGBDCodec` uses D as a residual-table index. Their byte meanings differ. [RGBD tokens and migration](RGBD_TOKENS.md) gives the exact layout and persistence examples.

## Operators

```python
from chromatensor.operators import chroma_add, chroma_matmul, LookupOperator, direct_grid_add

codec = UniformCodec(bits=4, minimum=0, maximum=2)
a = ChromaTensor.from_tensor(torch.tensor([0.2, 0.6]), codec)
b = ChromaTensor.from_tensor(torch.tensor([0.4, 0.8]), codec)
reference = chroma_add(a, b)
lookup = LookupOperator(codec, operation="add")
table_result = lookup(a, b)
integer_result = direct_grid_add(a, b)
```

`chroma_add`, `chroma_sub`, `chroma_mul`, `chroma_dot`, `chroma_matmul` and `chroma_activation` decode, compute numerically and encode. `chroma_similarity` and `chroma_distance` return numerical metrics. `LookupOperator` precomputes a numerical table and uses integer indexing during calls. `direct_grid_add` adds codes on matching nonnegative uniform grids and saturates at the largest code.

Direct grid addition chooses the codec's full-vocabulary storage dtype for its output. Valid operands in smaller signed integer containers can therefore produce codes beyond those containers' positive limits without wrapping.

`LearnedOperator` predicts a result ID with a floating-point embedding and MLP. Train it with `operator.fit(lookup, steps=..., seed=...)`; its returned list records training losses.

## Trainable layers

```python
from chromatensor.nn import ChromaMLP

model = ChromaMLP(1, [32, 32], 1, bits=8, representation="chroma")
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
inputs = torch.linspace(-1, 1, 256).reshape(-1, 1)
targets = inputs.sin()
optimizer.zero_grad()
predicted = model(inputs)
objective = (predicted - targets).square().mean() + model.auxiliary_loss()
objective.backward()
optimizer.step()
```

`ChromaLinear`, `ChromaEmbedding`, `ChromaActivation`, `ChromaLayerNorm` and `ChromaAttention` are in the same package. Quantization settings include `bits`, `representation`, `quantize_weights` and `quantize_activations`. The MLP accepts `latent_codes` to insert vector codebook bottlenecks and `latent_weight` for their commitment objective. Include `model.auxiliary_loss()` when training those bottlenecks.

`ChromaAttention` accepts `similarity_mode="dot_product"` or `"codebook_similarity"`. `ChromaAttention.from_torch` converts a supported ordinary multi-head attention module. Codebook similarity has no exact ordinary-attention conversion.

```python
from chromatensor.nn import convert_to_chroma, execution_report

ordinary = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
converted = convert_to_chroma(ordinary, bits=8, representation="chroma")
converted(inputs)
report = execution_report(converted)
restored = converted.to_torch()
```

Conversion copies supported layers and retains floating-point master weights. Converting back removes fake quantization and can change output values. The execution report records actual dtypes, code state bytes and remaining floating-point work.

For the optional lossless FP32 forward state, pass `bits=32, representation="rgbd"` to `ChromaQuantizer` or the supported neural layer/conversion API. The quantizer materializes actual four-channel weight and activation bytes before FP32 reconstruction. FP64 rounds to FP32. Master parameters, straight-through gradients, optimizer state and kernels remain floating point. Existing chroma8 behavior stays separate.

## Tensor and model files

```python
from chromatensor import save_ctensor, load_ctensor
from chromatensor.serialization import save_chroma_checkpoint, load_chroma_checkpoint

save_ctensor(cx, "artifacts/example.ctensor")
loaded = load_ctensor("artifacts/example.ctensor", device="cpu")
save_chroma_checkpoint(model, "artifacts/model.chroma.zip")
state = load_chroma_checkpoint("artifacts/model.chroma.zip")
model.load_state_dict(state)
```

`save_ctensor` returns the actual file size and accepts `compress=False` for raw payload comparison. Tensor files include codec tables, shapes, integer dtype, numerical output dtype and a checksum. Model checkpoints store state dictionaries; construct a matching model before loading them. They do not contain optimizer state or arbitrary architecture code.

The default checkpoint export preserves finite FP16, BF16 and FP32 state through RGBD. FP64 state uses numeric NPY members under manifest version 2. Complex state export raises `TypeError`. Passing a custom codec chooses its quantization rules for floating state entries.

All public loaders accept lower resource budgets.

```text
load_ctensor(path, device="cpu", *, max_file_bytes=None,
             max_header_bytes=None, max_payload_bytes=None)
TokenSequence.load(path, device="cpu", max_file_bytes=None,
                   max_header_bytes=None, max_payload_bytes=None)
load_chroma_checkpoint(path, model=None, device="cpu", *, max_file_bytes=None,
                       max_header_bytes=None, max_payload_bytes=None,
                       max_expanded_bytes=None, max_entries=10000)
ChromaCube.load(path, device="cpu", *, max_file_bytes=None,
               max_header_bytes=None, max_payload_bytes=None,
               max_expanded_bytes=None)
HolographicCube.load(path, device="cpu", *, max_file_bytes=None,
                    max_header_bytes=None, max_payload_bytes=None)
```

These are signature summaries. Set the keyword values in a normal call, for example `load_ctensor(path, max_file_bytes=8 * 1024**2, max_payload_bytes=16 * 1024**2)`. The standalone `chromatensor.holographic.load_hcube` accepts the same budgets as `HolographicCube.load`. [File formats and loader limits](FILE_FORMATS.md) defines the defaults, version compatibility, dtype policy and peak-memory boundary.

