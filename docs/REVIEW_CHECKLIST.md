# Review and repair checklist

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The current native hue checklist is in [TODO.md](../TODO.md). Its source is `0baada5d50c2fff69be4900a2b6820b714501b9573beada54059b0331bc7c6f9`, and all six [native acceptance gates](history/results/acceptance.json) pass. That revision passed 776 Python tests, 26 Rust tests on each platform, eight clean installed-wheel environments and the million-value integer reference. The sections below preserve earlier scalar RGBD repair evidence. Their release claims refer to source `69eb8fee...` and earlier revisions, archived under `results/history/before-native-hue/`; they do not certify native hue learning or execution.

Reviewed September 30, 2026. All fifteen original findings passed their repair checks on source `92ddfb4ad3d1a724f4373b2773e495b7abc086d0e09b3ab6ceb5acc44574146a`. Its ten-gate [acceptance receipt](history/results/history/before-rgbd-token-repair/acceptance.json) is historical. The new RGBD token contract changes executable source and requires fresh eleven-gate acceptance. [The repair report](REPAIR_REPORT.md) records the package, experiments and limits. [Full review](CODE_REVIEW.md) describes the initial evidence and [review-findings.json](history/results/review-findings.json) stores measured inputs and outputs.

## Historical scalar RGBD contract and evidence

The scalar RGBD executable source SHA-256 was `69eb8feebdc26d4e72bd1160ed7e5cf695d5035a65f62edebc1a061061ba1fba`. All eleven [archived acceptance gates](history/results/history/before-native-hue/acceptance.json) passed at `2026-10-01T04:34:20.484525Z`, with `complete=true`. The items below record that execution after the dashboard UTF-8 repair.

- [x] Verify actual default uint8 RGBD token rows and complete byte-only decoding, including signed zero and subnormal FP32 values. The suite includes 84 RGBD token tests.
- [x] Preserve explicit palette and residual IDs, and verify the documented `ChromaTokenizer(vocabulary_size=256)` migration.
- [x] Verify RGBD sequence and tensor views, row counts, safe `.ctensor` persistence and raw-byte decoder input.
- [x] Verify optional RGBD neural forward bytes and gradients with FP32 masters and kernels reported honestly. The suite includes 33 RGBD neural cases.
- [x] Pass the million-value independent NumPy byte oracle and cold-process decode without the source floats. Run `rgbd-tokens-20261001T041453876477Z-6e9c864ff2c9`, seed 7331, has zero bit mismatches and all four channel mutations change decoded values. Child PID 32344 decoded a 4,000,295-byte file containing 4,000,000 payload bytes, 251 header bytes and 44 container bytes. [Proof receipt](history/results/rgbd-token-verification.json).
- [x] Verify default RGBD inspector output and actual hexadecimal color labels separately from lookup IDs in the installed dashboard. The selected `COLOR_00FF00` has palette ID 2 and RGB `[0,255,0]`.
- [x] Pass the full host suite with zero skips on the scalar RGBD frozen source. All 702 tests passed in 75.94 seconds. [Archived test receipt](history/results/history/before-native-hue/test-verification.json).
- [x] Refresh all fourteen original defect probes and the separate seed-31 chroma8 training wrapper smoke on current source. All fourteen defect flags are false. Wrapper run `bench-20261001T042347877523Z-b33dfab1880b` completed.
- [x] Complete the current full pilot with all expected model and precision tuples. Run `bench-20261001T041728137087Z-970bb8116dbc` has 95 model records, 92 precision rows, 13 comparisons and 89 checkpoint replay probes. Final independent replay passed.
- [x] Complete cube `cube-20261001T041453929534Z-da641de6ef15` with six models, 16 geometry cases and 80 erasure cases. Internal runner checks and final independent replay passed.
- [x] Complete sustained run `release-research-20261001T042315415709Z-78aae1bc39ad` with twelve models, four seeds, 20 epochs each, disjoint 48,000/12,000 splits and all 10,000 final test observations. Ten-million-value actual packed-file checks passed its runner.
- [x] Independently replay all saved pilot, sustained-study and cube models and checkpoint-backed profiles in final acceptance.
- [x] Verify the exact new wheel across nine jobs with 27 installed CPU inference controls and nine RGBD token controls.
- [x] Save current-source installed-browser observations for exact RGBD inspector bytes, seven cube actions and all eleven views. RGBD observations. Original artifact `artifacts/dashboard/rgbd-observations.json` is omitted; regenerate it from source. and cube observations. Original artifact `artifacts/dashboard/cube-observations.json` is omitted; regenerate it from source. record the actual installed minimum wheel.
- [x] Validate the completed browser provenance envelope and pass all eleven acceptance gates.

The verified wheel has 157,119 bytes and SHA-256 `5caf5a93fb4dc8949d67415bb94030da06ed2f1adf528c7e240038b391eb0b04`. All nine jobs passed 683 CPU tests each with 19 CUDA-only skips and their installed proofs. Its 49 packaged code and browser assets match current source. The host suite executed the CUDA tests with zero skips.

The earlier RGBD source `cacea3870cefdb97cf167ceb07e2935d27df3ef6e0845854dc1e0bc7c1165277` passed 702 host tests and the million-token proof, pilot and cube checks before a browser check found mojibake in labels. Its receipts are historical under `results/history/before-rgbd-utf8-repair/`. The label repair changes source and requires fresh execution.

The checked items below preserve work completed before this token update. They do not certify the current executable revision. Earlier records and artifacts are under `results/history/before-rgbd-token-repair/` and `artifacts/review/before-rgbd-token-repair/`.

## Initial review evidence

- [x] Review every Python module and script, browser code, tests, and research documentation across independent reviewers.
- [x] Run the existing suite. 302 tests pass with zero skips.
- [x] Run separate CPU scalar, periodic, precision, operator, attention, and decoder smoke experiments.
- [x] Run CPU smoke models on official MNIST, CIFAR-10, and actual Tiny Shakespeare.
- [x] Verify the standalone training command.
- [x] Build a current-source wheel and check isolated package imports, cube inverse, finite Fourier decode, and bundled browser assets with host dependencies reused.
- [x] Exercise real Cube Lab creation, layer turn, exact displayed-row restoration, and partial spectral reconstruction in the browser.
- [x] Save fourteen defect reproductions, source hashes, configuration, and current evidence separately from canonical research records.

## Required repairs

- [x] R001, P1. Prevent future-token dependence in every causal activation quantizer. Prefix invariance passes for the actual language model and both attention modes. The fresh 95-record pilot reruns affected language and attention records.
- [x] R002, P1. Acceptance rejects skipped required tests, empty or inconsistent comparisons, absent training, stale source, changed artifacts, and mismatched packages. Negative tests also cover metadata headers and Git context. All ten final gates pass.
- [x] R003, P1. Retire the incomplete 0.1.0 wheel. Verify the exact 155,321-byte 0.2.0 wheel, SHA-256 `a950d2dac9d4b4e8f0b316cde5b22ad9e21d9949ba68ed795f277b7ef47fd674`, across nine independent environments. All 49 package assets match, and installed CPU inference and browser checks pass.
- [x] R004, P2. Preserve parameter aliases across unchanged parents and converted children in both conversion directions. Verify identity and optimizer parameter counts with quantization disabled.
- [x] R005, P2. Handle ordinary MHA through the supported adapter or reject unsupported generic conversion. Verify that intended quantizers execute inside converted Transformer structures.
- [x] R006, P2. Return every official MNIST observation for full-profile requests and redistribute subset quota shortages. Assert 60,000 train and 10,000 official test indices with full coverage.
- [x] R007, P2. Preserve learned centroid and neural parameter dtypes in configuration, token, and tensor-file roundtrips. Test FP64 and changed default dtype, or reject unsupported exports before writing.
- [x] R008, P2. Match Torch boolean indexing values and shapes. Test Python true/false, ellipsis, new axes, explicit slices, and advanced indexing.
- [x] R009, P2. Reject or correctly represent domains outside the learned centroid dtype. Test tiny positive domains, narrow intervals, and large finite endpoints.
- [x] R010, P2. Define useful binary residual refinement or reject two-symbol multistage configurations. Verify the supported precision-refinement contract.
- [x] R011, P2. Give cube runs immutable artifact directories and checkpoint hashes. An interrupted rerun must leave an earlier complete record's artifacts intact.
- [x] R012, P2. Validate allowed dashboard hostnames and exact origins. Foreign Host reads and computations must fail. Document the tested server-side scope.
- [x] R013, P2. Capture Git commit, dirty status, source/configuration hashes, dependency versions, and artifact hashes automatically. Bind verification to that evidence without inventing source provenance for older records.
- [x] R014, P3. Use a grid-add result container that holds every valid output code, or enforce canonical input containers. Cover accepted signed containers.
- [x] R015, P3. Reject degenerate logarithmic transformed bounds or normalize them stably. Adjacent positive endpoints must have defined finite behavior.

## Historical local release evidence

- [x] Execute the source-bound host suite. All 580 tests pass in 67.48 seconds with zero skips, including CUDA contracts.
- [x] Execute nine local Windows and Linux jobs for Python 3.11 through 3.14, including minimum dependencies. Each passes 562 CPU tests with only 18 CUDA-only skips. All dependency locks and installed-package checks pass. These are local jobs; hosted GitHub Actions was not run.
- [x] Verify independently installed dependencies in isolated Windows environments and Linux containers. No host dependency paths or user-site imports are reused. This establishes clean dependency installation, not newly provisioned physical hardware or driver installation.
- [x] Bind the exact source, wheel, checkpoints, stored predictions, and browser files by hashes and byte counts. Later result-only commits preserve valid evidence when those bytes match; original Git history stays unchanged.
- [x] Run 85 malformed-file and allocation-budget cases across all public loaders. [File formats and limits](FILE_FORMATS.md) define compatibility and dtype support.
- [x] Complete the 95-record pilot, 13 paired comparisons, 92 precision conditions, 89 held-out model replays, and two live CPU profiles.
- [x] Independently replay all 16 cube geometry cases, 80 erasure measurements, and six reconstruction models.
- [x] Complete twelve 20-epoch MNIST models across four seeds. Preserve disjoint 48,000/12,000 training/development partitions and evaluate 10,000 official test images after all new checkpoints are fixed. Earlier repository use of that test split remains disclosed.
- [x] Encode ten million FP32 values in actual packed files. RGB24 and fixed24 each use 30,000,000 payload bytes plus 131 metadata bytes and match exactly. RGBD preserves all sampled FP32 bit patterns.
- [x] Execute 27 installed CPU inference controls across the nine jobs, including 18 timed checkpoint/probe replays and nine finite integer-reference controls. Record actual precision and replay tolerances.
- [x] Verify seven installed-wheel Cube Lab interactions and all eleven views through the Computer Use API. Save actual rows, hashes, current checkpoint links, a native screenshot, and browser provenance. Original artifact `artifacts/review/before-rgbd-token-repair/cube-verification.json` is omitted; regenerate it from source.. No captured warnings or errors appeared.
- [x] Replay all fourteen original defect probes on final source and package. Every original defect flag is false in [repair-regressions.json](history/results/history/before-rgbd-token-repair/repair-regressions.json).
- [x] Run the separate chroma8 seed-31 smoke training command and preserve its own result.
- [x] Pass all ten gates in [acceptance.json](history/results/history/before-rgbd-token-repair/acceptance.json).

## Conditional claims outside this local release

- [ ] Establish a general benefit from learned color geometry. The tested matched controls show no color-label advantage. Four seeds and metadata-inclusive budgets do not establish superiority.
- [ ] Verify human screen-reader operation, public hosting, authentication, production traffic, or new physical hardware if those become release goals. The current local release does not claim those checks passed.

The original review repair checklist completed before the RGBD update. Current-source RGBD verification and all eleven acceptance gates are complete in the section above. Conditional scientific and deployment claims remain unverified. Earlier failed and successful records retain their original source identities and qualifications.
