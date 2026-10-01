# Experiment matrix

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](docs/RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The primary implementation is now the Rust hue tensor. Its operands and outputs are actual RGBD tokens. Native validation uses the finite domain `Z/391680Z` and equal 32-bit uint32 payload controls. The prior FP32 packing and neural experiments remain numerical comparisons, with archived release evidence.

## Native execution and release checks

All six native [acceptance gates](docs/history/results/acceptance.json) passed on source `0baada5d50c2fff69be4900a2b6820b714501b9573beada54059b0331bc7c6f9`. The host passed 776 Python tests with zero skips; its JUnit duration is 85.379 seconds. Both native builds passed 22 library tests and four CLI tests. All eight isolated installed-wheel environments passed. Earlier interim receipts retain their original identities.

| Check | Inputs and controls | Required evidence | Status |
| --- | --- | --- | --- |
| Native token mapping | All 391,680 valid phase positions, literal RGBD cases and invalid colors | Exact Rust encoder/decoder roundtrip and actual bytes | Passed |
| Tensor arithmetic | Broadcasting, matrices, vectors, sums, hue rotations and blend | Integer oracle equality, shape and empty cases, domain rejection | Passed |
| Ownership and inputs | Closed handles, concurrent reads and arithmetic, empty dimensions, oversized steps, NUL paths | Correct results or explicit errors, no invalid native lifetime | Passed |
| Persistence | Actual RGBD payload, corrupted channels, malformed headers, valid-checksum invalid colors | Exact readback, SHA-256 and byte-budget rejection | Passed |
| Million-token native proof | Seed 7331, one million uint32 phase positions per input, 32 physical bits per element | Exact RGBD and phase oracle matches, native operations, changed-byte effects and cold file decoder | Passed producer and replay |
| Measured CPU cost | Native operators and independent uint32 control on the same domain | Elapsed integer nanoseconds for three calls, explicit scope and actual payload/file bytes | Recorded, native slower on all seven operators |
| Synthetic retrieval | Defined generated color dictionary with eight templates | Deterministic retrieval and provenance, labelled synthetic | 256 of 256 queries correct, no learning claim |
| Rust builds | Windows x86-64 and pinned Linux x86-64 build | 22 library tests and four CLI tests per platform, actual artifacts | Passed both platforms |
| Native wheel installs | Windows and Linux, Python 3.11 through 3.14, eight clean environments | Exact wheel, isolated imports, neither PyTorch nor NumPy installed, executed native operations | Passed all eight |
| Final acceptance | Current tests, builds, proof, CLI and packages | Source and artifact hashes, every required native gate passed | All six gates passed |

The native proof producer is `python scripts/verify_native_hue.py`. It uses NumPy as an external integer reference. The native runtime package uses standard-library Python and the Rust library. Native payloads occupy four bytes per token, matching the uint32 control's four bytes. Complete native file size includes its fixed header, dimensions and checksum. Timing excludes input construction, validation and persistence, and includes operator calls, output allocation and release of earlier native outputs. The integer control includes dtype promotion to uint64 or int64.

[results/native-hue-verification.json](docs/history/results/native-hue-verification.json) records run `native-hue-20261001T052629909477Z-045185134a74`, one million values per input, seed 7331 and three repeated calls per operator. The 4,000,064-byte native file has 4,000,000 actual RGBD payload bytes and 64 overhead bytes. The uint32 control also has 4,000,000 payload bytes, with a separate 102-byte JSON descriptor. Their small container difference does not compress the payload. The isolated cold process loaded the saved file and checked bytes, phases and rotation without importing PyTorch or NumPy.

All seven operators matched the integer oracle. The separate 32 by 32 matrix contraction also matched. Changing a stored D byte changed that contraction, and changing canonical RGB bytes changed phase. Reshape, permutation and slicing preserved exact bytes.

| Operation on one million input values | Native total ms for three calls | uint32 control total ms for three calls |
| --- | --- | --- |
| Add | 122.993 | 27.588 |
| Subtract | 125.475 | 34.751 |
| Product | 133.837 | 28.563 |
| Rotate | 37.684 | 21.735 |
| Blend | 126.910 | 107.539 |
| Sum | 37.826 | 1.686 |
| Dot | 57.141 | 16.522 |

The native implementation is slower on every measured operator. These are aggregate elapsed times for three calls, not a latency distribution or an optimized kernel comparison. The data establishes representation correctness, with no performance advantage.

The synthetic dictionary used eight 32-position templates, seed 7332 and 256 queries. It retrieved all 256 correctly. Both representations used 1,024 template payload bytes. Native per-query construction, rotation, subtraction, copying and Python dictionary lookup took 22.062 ms. The control used a vectorized NumPy batch and the same lookup, taking 0.871 ms. Their execution structure differs. This is a defined dictionary task without training, neural inference or real-data generalization.

## Verified native packages

[results/native-hue-package-verification.json](docs/history/results/native-hue-package-verification.json) records all eight clean installs. Each environment checks isolated package paths, absent PyTorch and NumPy, native color encoding and decoding, actual D refinement, matrix arithmetic, shape operations and file roundtrip.

| Environment | Python | Result |
| --- | --- | --- |
| Windows 3.11 | 3.11.13 | Passed |
| Windows 3.12 | 3.12.13 | Passed |
| Windows 3.13 | 3.13.14 | Passed |
| Windows 3.14 | 3.14.6 | Passed |
| Linux 3.11 | 3.11.16 | Passed |
| Linux 3.12 | 3.12.14 | Passed |
| Linux 3.13 | 3.13.15 | Passed |
| Linux 3.14 | 3.14.7 | Passed |

The Windows wheel has 138,710 bytes and SHA-256 `5f0338cd1d8af65d2f5b8e3564d899fc031b33d9564942143d5f7c50c064584f`. The Linux wheel has 248,094 bytes and SHA-256 `0531040085baf3bc25645c389e46479350f6fc866eb78a3b65d6d2869ac90e2e`. These are separate `chromatic-hue` 0.1.0 platform wheels. Windows DLL SHA-256 is `b8333bd2ea87639ab577efbb1b79c336f92cb0bbd40865cbe9ded1a59aabd53e`; Linux SO SHA-256 is `fe17dd14284aea93e0a4fba137dccb3e3874be73776dca8d9bbff00a32081290`.

[results/acceptance.json](docs/history/results/acceptance.json) records all six native gates passed at `2026-10-01T05:26:53.099255+00:00`. The documented Python, C and Rust consumers also passed against final source and the Windows DLL, recorded in artifacts/native/documentation-final.json. Original artifact `artifacts/native/documentation-final.json` is omitted; regenerate it from source.. [TODO.md](TODO.md) closes required release work while keeping broader research claims unverified.

The color tokens represent the same modular phases as the uint32 control. A correct relabeling does not establish new mathematics or a numerical advantage. Synthetic retrieval does not establish training or real-data generalization. Measured CPU timings do not establish hardware or energy benefits.

## Historical numerical release

The previous source `69eb8feebdc26d4e72bd1160ed7e5cf695d5035a65f62edebc1a061061ba1fba` passed 702 host tests with zero skips in 75.94 seconds, nine installed-package jobs and eleven numerical acceptance gates. Its exact `chromatensor` 0.2.0 wheel had 157,119 bytes and SHA-256 `5caf5a93fb4dc8949d67415bb94030da06ed2f1adf528c7e240038b391eb0b04`. Those receipts are archived under [results/history/before-native-hue](docs/history/results/history/before-native-hue). They are historical evidence for the earlier implementation.

| Historical experiment | Provenance and scope | Finding |
| --- | --- | --- |
| FP32 RGBD byte proof | `rgbd-tokens-20261001T041453876477Z-6e9c864ff2c9`, one million finite FP32 bit patterns, seed 7331 | Zero byte and bit mismatches, separate-process decoding, 4,000,295 file bytes |
| Canonical pilot | `bench-20261001T041728137087Z-970bb8116dbc`, seeds 7 and 23, 95 records, 92 precision rows, 13 comparisons, 89 checkpoint probes | Chroma8/INT8 and RGB24/fixed24 controls had zero paired metric deltas |
| Sustained MNIST | `release-research-20261001T042315415709Z-78aae1bc39ad`, seeds 7, 23, 41 and 59, twelve models, 20 epochs, 48,000/12,000 train/development splits, final 10,000 official tests | FP32 mean accuracy 0.975175; INT8 and chroma8 each 0.976075 with identical logits and curves |
| Large numerical files | Ten million seeded FP32 uniform values in [-1,1], seed 7331 | RGB24/fixed24 had identical 30,000,000-byte payloads; RGBD preserved sampled FP32 bits in 40,000,000 payload bytes |
| Cube geometry and reconstruction | `cube-20261001T041453929534Z-da641de6ef15`, 16 geometry cases, 80 erasure cases, six models, 24 epochs, 1,024 synthetic training fields and 256 held-out fields | Direct RGB mean MSE 0.003196234; spectral chromatic and numeric controls each 0.01138765 |

The earlier pilot used official MNIST and CIFAR-10 subsets with 12,000 training and 2,000 test images per seed. Tiny Shakespeare used 2,048 training and 512 held-out windows. Image and language models trained for six epochs; scalar and periodic models trained for 24. Its four precision distributions each used one million values. Those real datasets retain their names and provenance. The new native synthetic dictionary is a separate experiment.

The sustained MNIST procedure opened the official test split after all twelve checkpoints were fixed. Earlier repository runs had already used that split. This procedure's withholding did not make it historically unseen. The quantized models used floating masters and fake quantization. Their agreement does not demonstrate integer-only color training or a storage benefit.

The historical numerical file check measured RGB24/fixed24 MAE `2.980352106067395e-08` and maximum error `5.96046483281043e-08`. Those files each had 131 metadata bytes. RGBD had 101 metadata bytes. Whole-process peak RSS was 2,816,032,768 bytes, with an observed experiment increase of 1,266,499,584 bytes. These counts describe that numerical process and do not measure the new native tensor.

The older cube fields used complex128 Fourier operations with byte phase/magnitude storage and numerical scales. Spectral chromatic and numerical polar results agreed exactly. They did not recover arbitrary erased tensors or execute the native hue ring.

## Earlier review and evidence revisions

The original September 30 review reproduced fourteen defects beyond its existing tests and tracked fifteen repairs. The old quantized causal models had a range-estimation defect that read future positions. The repair derives activation ranges from each token's own features. Older affected language results retain that qualification.

The pre-token-update source passed 580 tests and ten gates, preserved under `results/history/before-rgbd-token-repair/`. A later RGBD revision passed interim runs before browser inspection found UTF-8 mojibake. Those records remain under `results/history/before-rgbd-utf8-repair/`. The corrected `69eb8f` numerical release completed before the native hue task.

[The legacy results](RESULTS.md), [measurement guide](docs/MEASUREMENT.md), [cube measurements](docs/CUBE_RESULTS.md), [code review](docs/CODE_REVIEW.md) and [research log](RESEARCH_LOG.md) retain their full account. `python scripts/verify.py --legacy --output results/legacy-acceptance.json` selects the eleven numerical gates and writes a separate report. Current native acceptance must use fresh Rust, Python, binary and package identities.


## Public release candidate

The frozen source identity is `2be972ac122e0b2bb178064370420343bf91e402745bffeddf47816779ee3957`. Locked Windows Python3.12 CPU and an independent clean clone each passed 823 tests with 19 CUDA-only skips. The separate host CUDA diagnostic passed 842 tests but used dependency versions outside supported floors. Eight native installed jobs and one selected locked research job passed. See [the release audit](docs/RELEASE_AUDIT.md) for exact environments.

The candidate native proof uses 1,000,000 seeded phase tokens, seed7331, three timing repeats and 32 physical bits per value for both native RGBD and uint32 controls. Rust integer operators match the independent oracle exactly. Its saved file has 4,000,000 payload bytes and 64 overhead bytes. The uint32 control has the same payload and 102 metadata bytes. Cold-process loading uses only the saved native file, without Torch or NumPy. Native arithmetic is finite-ring execution; legacy training and Fourier calculations remain floating point.

| Operator | Native total ms, 3 calls | uint32 control total ms, 3 calls | Exact output |
| --- | --- | --- | --- |
| add | 184.6703 | 70.8614 | yes |
| subtract | 324.5087 | 91.4743 | yes |
| multiply | 332.5115 | 76.6445 | yes |
| rotate | 104.0229 | 57.1374 | yes |
| blend | 403.4195 | 313.1474 | yes |
| sum | 82.2990 | 5.3726 | yes |
| dot | 136.9390 | 36.4756 | yes |

Timings describe this local run during release verification. They are not optimized hardware claims. Native operators were slower for every measured operation. Both stored representations use the same payload budget. The domain has 391,680 symbols, so four bytes do not imply 32 effective information bits.

The final legacy smoke produced 17 model records, 92 precision rows using 4,096 values per distribution and 4 operator rows. The cube run used seed7, two epochs, 3 models, 8 exact geometry cases and 40 erasure measurements. These synthetic tests are named separately from actual MNIST, CIFAR and Tiny Shakespeare experiments. The full canonical datasets and historical trained checkpoints were not rerun. Historical numbers above remain unchanged.

## Static documentation checks, 2026-10-01

The documentation build renders 47 Markdown documents from the public release selection. Thirty release-selection, privacy-scanner and architecture tests passed. The architecture check validated 26 components, 40 relationships and 140 inventoried files. All 202 source-document links resolved locally. The hosted draft loaded the user guide and architecture explorer; searching RGBD token reduced the map to one component. These are documentation and hosting checks, not new numerical experiment results.
