# Public release status

The local source release is ready to publish within the tested scope. [The checklist](RELEASE_CHECKLIST.md) and [audit](docs/RELEASE_AUDIT.md) record the completed gates; the handoff provides final commit/archive identities and scan receipts. Static documentation publication is tracked in [the hosting guide](docs/HOSTING.md). Full research-matrix/canonical replay, Linux C consumer and broader platforms remain unexecuted research work, with no current pass claim.

## Visual explanation page

- [x] Add a labeled experimental network animation and exact hue-byte illustration.
- [x] Verify JavaScript mapping, network reference outputs and mobile controls.
- [x] Add social links, share metadata, a PNG card and documentation navigation.

## Historical native checklist

# Acceptance checklist

The current task replaces the primary numerical tensor with a native hue tensor. All six native [acceptance gates](docs/history/results/acceptance.json) passed at `2026-10-01T05:26:53.099255+00:00` on source `0baada5d50c2fff69be4900a2b6820b714501b9573beada54059b0331bc7c6f9`. The earlier eleven-gate numerical release is archived under [results/history/before-native-hue](docs/history/results/history/before-native-hue). It does not certify the new Rust library.

## Implemented native behavior

- [x] Store actual canonical RGBD tokens in an owned Rust tensor, with no retained floating tensor or decoded phase buffer.
- [x] Implement the hue encoder and decoder in Rust. D refines a hue step; it is separate from the legacy FP32 exponent format.
- [x] Make `Tensor`, `HueTensor` and `ColorTensor` the primary API. Keep numerical comparison APIs explicit and lazily imported.
- [x] Implement modular addition, subtraction, product, matrix multiplication, dot, sum, rotation and shortest-arc blend with integer operators.
- [x] Implement scalar and trailing-axis broadcasting, reshape, permutation and bounded positive-step slices.
- [x] Enforce canonical colors, finite phase bounds, matching shapes, rank and allocation budgets before native input copies.
- [x] Add portable RGBD persistence, bounded loading, SHA-256 corruption checks and adjacent temporary writes.
- [x] Expose a C ABI and a standard-library Python binding. Keep native handles alive during concurrent operations and `close()`.
- [x] Provide a standalone CLI that encodes, decodes, stores and computes without Python.
- [x] Build separate Windows and Linux `chromatic-hue` platform wheels without PyTorch or NumPy runtime dependencies.
- [x] Document the changed arithmetic, color projection, actual storage costs, ownership, migration and operating instructions in [the native guide](docs/NATIVE_HUE.md).

## Final verification

- [x] Pass the complete host suite after the final native source freeze. All 776 cases passed with zero skips; JUnit records 85.379 seconds. [Test receipt](docs/history/results/test-verification.json).
- [x] Pass the 22 Rust library cases and four CLI cases on each of [Windows](docs/history/results/native-hue-build-windows-x86_64.json) and [Linux](docs/history/results/native-hue-build-linux-x86_64.json).
- [x] Verify one million native RGBD tokens at seed 7331 against an independent integer oracle. All seven measured operators and a 32 by 32 contraction agree exactly; actual mutations change results, and separate-process decoding preserves bytes and phases.
- [x] Record equal 32-bit uint32 control payloads, actual native file bytes and elapsed integer nanoseconds for three repeated calls. Native operators are slower on every measured operation. No speed advantage is claimed.
- [x] Verify the final standalone executable and source-bound CLI proof, including persisted D-byte mutation driving matrix arithmetic. The executable SHA-256 is `833b191321731423c3eebde9d36828f57ecd86b90a5a009db6ff2f5e43bd56fa`.
- [x] Install the exact platform wheels in eight clean Python 3.11 through 3.14 environments on Windows and Linux. Every environment passed and confirmed neither PyTorch nor NumPy is installed.
- [x] Pass all six native gates in [results/acceptance.json](docs/history/results/acceptance.json), with current source and artifact identities.
- [x] Execute the documented Python, C and Rust examples against the final Windows library. Source-bound consumer receipt. Original artifact `artifacts/native/documentation-final.json` is omitted; regenerate it from source..
- [x] Update this checklist, [the experiment matrix](EXPERIMENT_MATRIX.md) and [research log](RESEARCH_LOG.md) from completed receipts.

The native million-token receipt is [results/native-hue-verification.json](docs/history/results/native-hue-verification.json), run `native-hue-20261001T052629909477Z-045185134a74`. Its file contains 4,000,000 payload bytes and 64 metadata/checksum bytes. The package matrix is [results/native-hue-package-verification.json](docs/history/results/native-hue-package-verification.json). Windows wheel size is 138,710 bytes with SHA-256 `5f0338cd1d8af65d2f5b8e3564d899fc031b33d9564942143d5f7c50c064584f`; Linux wheel size is 248,094 bytes with SHA-256 `0531040085baf3bc25645c389e46479350f6fc866eb78a3b65d6d2869ac90e2e`. Earlier interim receipts remain diagnostics and do not replace this evidence.

## Recorded repairs

The native work repaired a Windows library/executable PDB name collision, constructor validation before allocation, concurrent handle lifetime, empty-array shape preservation, large slice-step conversion and embedded NUL file paths. [RESEARCH_LOG.md](RESEARCH_LOG.md) records each failure and repair. Final receipts bind the corrected source.

## Remaining research limits

- [ ] Establish native hue GPU execution with actual device evidence.
- [ ] Implement and verify autograd and native hue neural training if that work is requested.
- [ ] Establish useful learning behavior on real datasets with appropriate controls.
- [ ] Establish optimized hardware, throughput, memory or energy benefits against equal-storage controls.

These items are unverified research directions. The finite hue ring is established mathematics. Ring multiplication is not physical color mixing, and hue operations do not preserve arbitrary real-number algorithms. The native token uses four physical bytes, so no compression benefit is claimed.

## Hardware story and public media

- [x] Add the maintainer's story and an authorized LED assembly clip, with the camera enclosure described as planned.
- [x] Verify full video decoding, preserved AAC audio and removal of identifying metadata.
- [x] Check exact-path video release selection. All 31 release-selection, privacy-scanner and architecture tests passed.
- [ ] Add a reviewed transcript or captions. Local transcription was not reliable enough to publish.
- [ ] Build the camera enclosure and measure readback errors under controlled lighting.
