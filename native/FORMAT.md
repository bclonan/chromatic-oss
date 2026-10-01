# Native hue tensor format

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](../docs/RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The Rust library stores actual RGBD tokens. Each token has four unsigned bytes. R, G and B identify one of the 1,530 saturated colors around the RGB hue wheel. D divides the next color step into 256 parts. A token decodes to the integer phase `hue_index * 256 + D`, in the domain `0..391680`.

This format differs from the earlier FP32 RGBD packing. D is a hue subdivision here. The file contains no FP32 values or copy of an input numerical tensor.

Version 1 uses this layout. Length fields use little-endian byte order.

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 8 | Magic `48 55 45 43 54 00 0d 0a` |
| 8 | 2 | Version, `1` |
| 10 | 2 | Reserved flags, `0` |
| 12 | 4 | Rank, at most 32 |
| 16 | 8 | Token count |
| 24 | `8 * rank` | Dimensions |
| `24 + 8 * rank` | `4 * count` | Token rows in R, G, B, D order |
| `24 + 8 * rank + 4 * count` | 32 | SHA-256 of all preceding bytes |

A scalar has rank zero and one token. A shape with a zero dimension has no tokens. Dimensions must multiply to the declared token count without overflowing the address space. The library also limits a tensor to 67,108,864 tokens. A two by two tensor takes 88 file bytes, including its 16 token bytes.

`load_tensor(path, max_bytes)` checks the open file's size against the caller's budget before allocating tokens. It rejects incorrect magic, unsupported versions, nonzero flags, invalid ranks, inconsistent shapes or lengths, checksum failures, trailing bytes and colors outside the canonical hue wheel. SHA-256 detects changes to a file. It does not identify who wrote the file.

`save_tensor` writes a new temporary file beside the destination, syncs it, then renames it. A failed write leaves an existing destination unchanged. The final rename follows the operating system and filesystem's rules.

## Standalone commands

Build the native executable with `cargo build --release --manifest-path native/hue_tensor/Cargo.toml`. The executable is `native/hue_tensor/target/release/hue-tensor-cli`, with `.exe` on Windows.

```text
hue-tensor-cli encode 2x2 0,256,131072,391679
hue-tensor-cli decode 2 "255,0,0,0;255,1,0,255"
hue-tensor-cli save left.hue 2x2 256,65536,131072,196608
hue-tensor-cli save right.hue 2x2 32768,98304,262144,390000
hue-tensor-cli matmul left.hue right.hue result.hue
hue-tensor-cli read result.hue 1048576
hue-tensor-cli demo proof.hue
```

The commands print the actual RGBD rows and their decoded integer phases as JSON. `read` runs without an encoder input or Python process. `demo` checks matrix multiplication against a separate integer phase calculation, changes the stored D byte, checks the changed result, then saves and reloads that result. An optional path keeps the proof file for a separate `read` process.

The executable also supports `add`, `sub`, `product`, `rotate`, `blend` and `sum`. Addition, subtraction, products and matrix contraction use arithmetic modulo 391,680. Rotation adds phase ticks around the hue wheel. Blend follows the shortest hue arc, using a byte weight. These operations do not promise the behavior of real-number tensor arithmetic.
