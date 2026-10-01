# RGBD bytes are the tokens

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

This page documents legacy scalar RGBD, where D holds an FP32 exponent. The primary native hue tensor uses another contract, where D refines hue position and Rust executes the finite hue algebra. Use [the native hue guide](NATIVE_HUE.md) for that implementation. The scalar RGBD release receipts here belong to the archived source before the native hue revision.

`ChromaTokenizer()` stores one actual `[R, G, B, D]` byte row for each numerical value. Its default sequence contains a `torch.uint8` tensor with shape `[N, 4]`, where `N` is the number of values. Decoding uses those bytes and the saved shape and codec metadata. The sequence retains no original floating tensor or fitted palette.

This default changes the tokenizer API in the 0.2.0 update. Use `ChromaTokenizer(vocabulary_size=256)` to keep the earlier bounded palette-ID behavior. Existing explicit vocabularies and residual configurations still use integer IDs.

## The fourth byte matters

RGBD rearranges the finite IEEE-754 FP32 bit pattern into four color bytes. The high bit of R stores the sign. The remaining seven bits of R, all eight bits of G, and all eight bits of B store the 23 fraction bits. D stores the eight-bit biased exponent. D is numerical exponent metadata, not image depth.

| Value | R | G | B | D |
| --- | --- | --- | --- | --- |
| `1.0` | 0 | 0 | 0 | 127 |
| `2.0` | 0 | 0 | 0 | 128 |
| `-1.0` | 128 | 0 | 0 | 127 |

The RGB preview of `1.0` and `2.0` is the same black color. D distinguishes their numerical values. Keep all four bytes when saving or decoding a token. A `COLOR_RRGGBB` label describes its three display channels and cannot replace the complete RGBD row.

The decoder reconstructs the FP32 word with this integer expression:

```text
((R & 128) << 24) | (D << 23) | ((R & 127) << 16) | (G << 8) | B
```

Finite FP32 values round-trip bit for bit, including signed zero and subnormal values. D=0 represents zero or a subnormal value. D=255 represents nonfinite IEEE values and is rejected. RGBD occupies 32 payload bits per scalar before shape, codec, checksum and container costs. It rearranges FP32 bits without compressing them.

FP16 and BF16 values can widen to FP32 without losing their numerical values. FP64 inputs round to FP32 before encoding. Recording FP64 as the output dtype only widens that reconstructed value. Encoding rejects nonfinite inputs, finite values that overflow FP32, and nonzero values that underflow to zero.

## Encode, inspect and decode

```python
import torch
from chromatensor import ChromaTokenizer, ChromaDecoder, TokenSequence

x = torch.tensor([1.0, 2.0], dtype=torch.float32)
sequence = ChromaTokenizer().encode(x)

assert sequence.tokens.dtype == torch.uint8
assert sequence.tokens.tolist() == [[0, 0, 0, 127], [0, 0, 0, 128]]
assert sequence.num_tokens == 2
assert sequence.storage_nbytes == 8
assert sequence.rgbd.shape == (2, 4)
assert sequence.rgb.shape == (2, 3)

decoded = ChromaDecoder().decode(sequence)
assert torch.equal(decoded.view(torch.int32), x.view(torch.int32))

# Raw byte rows also decode without a vocabulary.
assert torch.equal(ChromaDecoder().decode(sequence.rgbd), x)
```

`sequence.tokens` has flat rows. `sequence.rgbd` restores the original numerical shape and appends four channels. `sequence.rgb` exposes the first three channels for display. `num_tokens` counts rows, so four stored bytes still form one token. `storage_nbytes` counts token and residual-length buffers, excluding Python objects and serialized metadata. `sequence.to(device)` moves its integer buffers.

`ChromaTensor.from_tensor(x)` also defaults to RGBD. Its `codes` and `rgbd` contain the actual four bytes with the original value shape. Accessing `.rgbd` on a tensor or sequence that uses a different codec raises `ValueError`. `ChromaDecoder()` accepts raw `uint8` tensors with a trailing four-channel dimension and returns FP32 values.

## Save bytes and load them in another process

```python
path = "artifacts/rgbd-tokens.ctensor"
file_bytes = sequence.save(path, compress=False)

loaded = TokenSequence.load(
    path, device="cpu", max_file_bytes=8 * 1024**2,
    max_header_bytes=256 * 1024, max_payload_bytes=4 * 1024**2,
)
assert torch.equal(ChromaDecoder().decode(loaded), x)
```

`TokenSequence.save` writes the RGBD rows through the checksummed `.ctensor` format. `compress=False` keeps the byte payload raw by default. `TokenSequence.load` reads that file with the public loader's resource budgets and rejects a non-RGBD tensor file. Neither call requires the encoder object or original values. These convenience methods apply to RGBD sequences. Explicit palette and residual sequences retain their existing in-memory codec metadata and ID contracts.

The independent proof command is `python scripts/verify_rgbd_tokens.py`. Current run `rgbd-tokens-20261001T041453876477Z-6e9c864ff2c9` checked one million finite FP32 bit patterns at seed 7331, including nine known prefixes. Every byte matched an independent NumPy oracle, and reconstruction had zero bit mismatches. Mutating each channel changed decoded values. Child PID 32344 decoded only the saved `.ctensor` bytes and expected hash, without the original tensor.

The scalar proof file contains 4,000,000 payload bytes, 251 JSON header bytes and 44 container bytes. Total size is 4,000,295 bytes, or 32.00236 stored bits per value. [The archived source-bound proof receipt](history/results/history/before-native-hue/rgbd-token-verification.json) records source `69eb8feebdc26d4e72bd1160ed7e5cf695d5035a65f62edebc1a061061ba1fba` and the file hashes. All eleven [archived acceptance gates](history/results/history/before-native-hue/acceptance.json) passed, including independent replay of this scalar token proof. These receipts are historical.

The installed minimum wheel also displayed `[1, 2, -1, 1.5]` as the exact rows `[0,0,0,127]`, `[0,0,0,128]`, `[128,0,0,127]` and `[64,0,0,127]`. Measured MAE was zero. Selecting the second token displayed D equal to 128 and decoded value 2. The codebook separately displayed the actual hexadecimal label `COLOR_00FF00`, palette ID 2 and RGB `[0,255,0]`. Browser observations. Original artifact `artifacts/dashboard/rgbd-observations.json` is omitted; regenerate it from source. record the current source and wheel hashes, with no console warnings or errors. [The inspector screenshot](../artifacts/dashboard/rgbd-current.jpg) shows the RGBD interaction. The combined release envelope check passed.

## Choose a palette or residual IDs explicitly

```python
from chromatensor import ChromaTokenizer

palette = ChromaTokenizer(vocabulary_size=256, minimum=-1, maximum=1)
ids = palette.encode(torch.tensor([-0.5, 0.5]))
assert ids.tokens.ndim == 1

residual = ChromaTokenizer(vocabulary_size=16, levels=3, tolerance=0.001)
refined = residual.encode(torch.linspace(-1, 1, 1000))
```

Passing a vocabulary, vocabulary size, numerical bounds, multiple levels or tolerance selects the explicit palette/residual path. `ChromaVocabulary.symbol(id)` formats the actual palette RGB bytes as hexadecimal `COLOR_RRGGBB`. The integer ID remains a separate lookup index. This path quantizes values according to its bounds and vocabulary. It does not inherit default RGBD's lossless FP32 guarantee.

`LearnedRGBDCodec` has a separate format. Its RGB channels identify a coarse learned palette value, and D is a residual-table index. Its D byte has no exponent meaning. Always decode bytes with the codec recorded in their metadata.

## Neural forward state

The optional neural representation `representation="rgbd", bits=32` materializes four bytes per encoded weight or activation value during its forward pass. It then reconstructs FP32 values for ordinary PyTorch kernels. Inspection reports must count those actual bytes separately from live parameters and temporary tensors.

RGBD forward state preserves finite FP32 values. FP64 values pass through FP32 rounding. Training keeps floating-point master parameters, optimizer state and straight-through gradients. Matrix products and activations still run through floating-point kernels. A byte-valued forward state does not establish integer-only neural execution. Existing chroma8 and matched INT8 controls retain their original quantization levels and comparison contract.

```python
from chromatensor.autograd import ChromaQuantizer

quantizer = ChromaQuantizer(bits=32, representation="rgbd")
state = quantizer.encode(torch.tensor([1.0, 2.0]))
assert state.rgbd.tolist() == [[0, 0, 0, 127], [0, 0, 0, 128]]
assert state.codes is state.rgbd
assert state.payload_bytes == 8
assert state.range_metadata_bytes == 0
```

After a forward call, `last_state.rgbd` records its actual byte buffer. The state counts the aliased `codes` and `rgbd` storage once. Module execution reports name this format `RGBD uint8 quadruples` and keep payload bytes, retained bytes and master-parameter bytes separate. Current host verification includes 33 RGBD neural cases. Converting ordinary layers leaves unsupported ordinary nonlinearities such as `torch.nn.Tanh` unchanged.
