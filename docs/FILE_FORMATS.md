# File formats and loader limits

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

Native `HueTensor.save` and `HueTensor.load` use a separate bounded, checksummed binary RGBD format with an explicit version, modulus and shape. See [the native format contract](NATIVE_HUE.md). Its D byte means fine hue position. The JSON and NumPy formats below belong to the legacy numerical controls and cannot load native hue files.

ChromaTensor loads fixed JSON schemas, integer code buffers and numeric NumPy arrays. It does not load pickled Python objects. A checkpoint contains a state dictionary, so the caller must construct its model architecture before applying that state.

## Resource budgets

All four public loaders accept `max_file_bytes`, `max_header_bytes` and `max_payload_bytes` as keyword arguments. `None` selects the default. Byte limits must be nonnegative integers. A caller can lower a limit; a larger value does not raise the format ceiling. Missing files and other file access failures retain the usual Python file exceptions. Invalid metadata, failed integrity checks and exceeded budgets raise `ValueError`.

One MiB is 1,048,576 bytes. One GiB is 1,073,741,824 bytes.

| Loader | Default file limit | Default header limit | Default payload limit | Additional limits |
| --- | --- | --- | --- | --- |
| `load_ctensor` | 1 GiB + 17 MiB | 16 MiB | 1 GiB | None |
| `load_chroma_checkpoint` | 1 GiB + 17 MiB | 16 MiB | 1 GiB | Expanded ZIP entries total 1 GiB + 16 MiB, at most 10,000 members |
| `ChromaCube.load` | 1 GiB + 17 MiB | 16 MiB | 1 GiB | Expanded ZIP entries total 1 GiB + 16 MiB, exactly four named members |
| `HolographicCube.load` and `load_hcube` | 1 GiB + 2 MiB | 1 MiB | 1 GiB | None |

For `.ctensor` and `.hcube`, lowering the header or payload limit also lowers the file ceiling to their sum plus one MiB. `max_file_bytes` can make that ceiling smaller. The extra MiB allows file framing and zlib overhead.

For ZIP containers, `max_header_bytes` caps the central directory before `ZipFile` builds its member records. It also caps JSON and embedded tensor headers. `max_expanded_bytes` caps the sum of the members' uncompressed file sizes before any member opens. Its default is the selected payload limit plus the selected header limit. `load_chroma_checkpoint` also accepts `max_entries`, an integer from 1 through 10,000. That count includes `manifest.json`. Cube archives have four fixed members and do not accept a different entry count.

Payload means different buffers in each format.

- A `.ctensor` payload is its decompressed integer code buffer.
- A `.hcube` payload is its decompressed three-byte coefficient buffer.
- A `.ccube` payload budget covers the embedded tensor's integer code buffer and its int64 cell identities together.
- A checkpoint budget caps the sum of decoded state tensor bytes. Each embedded `.ctensor` code buffer must also fit the remaining budget before decoding. For FP16 and BF16 state entries, the four-byte RGBD buffer can be larger than the decoded state tensor.

Choose limits for the file you expect to load.

```python
from chromatensor import ChromaCube, HolographicCube, load_ctensor
from chromatensor.serialization import load_chroma_checkpoint

KiB = 1024
MiB = 1024 * KiB
limits = dict(max_file_bytes=8 * MiB, max_header_bytes=256 * KiB,
              max_payload_bytes=16 * MiB)

tensor = load_ctensor("artifacts/example.ctensor", **limits)
state = load_chroma_checkpoint("artifacts/model.chroma.zip", **limits,
                              max_expanded_bytes=16 * MiB, max_entries=256)
cube = ChromaCube.load("artifacts/example.ccube", **limits,
                      max_expanded_bytes=16 * MiB)
field = HolographicCube.load("artifacts/example.hcube", **limits)
```

These budgets bound files, metadata and payload expansion. They are not peak RAM limits. Loading can retain compressed bytes, decompressed bytes, array copies, learned codec parameters and validation scratch tensors at the same time. A device transfer adds another copy. CPU validation can widen integer codes to int64. Choose a smaller payload budget when the process has little memory, and use an operating system process limit when you need a hard RAM or CPU limit.

## Tensor files

`.ctensor` retains format version 1. A file contains the eight-byte `CTENSOR` version marker, a four-byte little-endian header length, a UTF-8 JSON header, an integer payload and a 32-byte SHA-256 digest. The payload is raw or zlib-compressed.

The header records logical shape, code shape, code dtype, decoded dtype, codec configuration and byte counts. Supported code dtypes are uint8, uint16, uint32, uint64, int8, int16, int32 and int64. The decoded dtype is float16, bfloat16, float32 or float64. Code validation still checks the selected codec's symbol rules. An integer container does not increase the codec's vocabulary.

Default RGBD preserves finite FP32 values. A float64 input encoded with RGBD rounds to FP32 before storage. The encoder rejects values that overflow FP32 and nonzero values that round to zero. Recording float64 as the decoded dtype does not recover discarded precision. Use a codec whose numerical behavior fits the task and account for its reconstruction error.

The default tokenizer writes these same actual four-byte RGBD tokens. `TokenSequence.save(path, compress=False)` stores the uint8 rows through `.ctensor`, including logical shape, decoded dtype and RGBD codec metadata. `TokenSequence.load(path, **budgets)` forwards device and resource limits to `load_ctensor` and rejects a tensor file with another codec. Loading needs no tokenizer object, palette or original numerical array. These methods apply to RGBD sequences; explicit palette/residual sequences use their separate ID contract. [RGBD tokens](RGBD_TOKENS.md) specifies the sign, fraction and exponent byte layout.

Learned configurations now include centroid or neural parameter dtypes. Saved files and token sequences preserve those dtypes independently of `torch.set_default_dtype`. Older learned configurations without dtype fields load their parameters as FP32. The version marker stays unchanged because the codec metadata carries this information.

## Model checkpoints

Checkpoint manifests use version 1 or 2. Both store JSON and named data members in a ZIP archive.

| State dtype under the default export | Stored member | Manifest version | Numerical behavior |
| --- | --- | --- | --- |
| Finite float16, bfloat16 or float32 | RGBD `.ctensor` | 1, or 2 when the same checkpoint also has FP64 state | Lossless |
| Finite float64 | Numeric float64 `.npy` | 2 | Lossless |
| Boolean or supported integer tensor | Integer `.npy` | 1 or 2 | Lossless |
| Complex, sparse or unsupported floating tensor | Export rejects it | None | No file replaces the destination |

The default means `save_chroma_checkpoint(model, path)` with no codec argument. If the caller supplies `codec=...`, floating state entries use that codec, including FP64 entries. The caller has chosen its clipping, quantization and precision rules. A custom RGBD codec can therefore round FP64 state; a bounded codec can clip or quantize weights. Custom export does not carry a general lossless guarantee.

The loader retains support for version 1 checkpoints. Version 2 adds the `numeric` entry kind for float64 NPY arrays. An older loader that supports only version 1 will reject a version 2 checkpoint. NPY format versions 1 and 2 are supported. The loader checks array dtype, shape, byte count and budget before copying its numeric data. It accepts integer arrays with either byte order and reads the logical values of Fortran-ordered arrays. Object arrays and pickled payloads are forbidden. Numeric FP64 entries must be finite.

Loading with `model=...` first validates and builds the complete state dictionary. It then calls `model.load_state_dict`. File validation failure does not apply a partial state to the model. PyTorch model compatibility errors remain the caller's responsibility.

## Cube files

`.ccube` remains a version 1 ZIP format. Its four members are `tensor.ctensor`, `cell_ids.bin`, `manifest.json` and `manifest.sha256`. The manifest records shape, axis names, identity dtype and payload checksums. Cell identities are little-endian int64 values. Loading checks that they form a permutation of the original row-major positions.

`.hcube` remains a version 1 checked binary format. It has an eight-byte marker, a four-byte JSON-header length, a raw or zlib coefficient buffer and a 32-byte SHA-256 digest. Each coefficient uses one phase byte and two magnitude bytes. The header stores a finite amplitude scale, axis names and the fixed phase, magnitude and FFT conventions. Decode uses complex128 and float64 Fourier operations. The file represents software coefficients, not optical storage.

## Rejected files and remaining boundaries

The loaders reject duplicate JSON keys, malformed schemas, unsupported versions, impossible byte counts, invalid shapes, truncated data and invalid compressed streams. Bounded decompression rejects streams that expand beyond the declared payload, contain trailing compressed data or fail to finish. ZIP loaders reject duplicate, missing and unknown members, encrypted members and compression methods other than stored or deflated. They do not extract paths onto the filesystem.

SHA-256 checks detect accidental corruption in tensor and Fourier files and compare cube manifest and payload digests. ZIP CRC checks cover checkpoint member corruption. These checks do not authenticate the sender. Someone who edits a file can also recompute its checksums. They do not prove that a model or learned codec reconstructs useful values.

The file tests cover explicit budgets and forged metadata, compression bombs, duplicate members, object NPY arrays, truncation and dtype roundtrips. They do not establish exhaustive fuzz coverage, parser CPU limits, maximum-size throughput, cross-endian machine testing or a guarantee that any chosen budget fits available RAM.
