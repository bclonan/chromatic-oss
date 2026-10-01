# Code, architecture, and software review

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

This report describes the pre-repair source and measurements. The autonomous repair work follows [the current checklist](REVIEW_CHECKLIST.md). [Repair regression evidence](history/results/repair-regressions.json) replays the original review inputs. Historical runs remain under `results/history/`; this report does not attribute later repairs to the old source.

Reviewed September 30, 2026, America/New_York. The reviewed source commit is `cce9742ef9b15d85e29fccecfc8401240b7aaed5` on `codex/chromatensor-system`. The review adds documentation and diagnostic evidence. The library and experiment implementations retain their reviewed behavior.

ChromaTensor is a working research library. It stores actual integer codes, trains actual models, reads real image and language datasets, writes checkpoints, and serves a dashboard backed by Python computations. The code supports the narrow representation claims. Full acceptance remains open. I found 15 actionable issues, including future-token leakage in causal models, an acceptance script that accepts skipped tests, and a committed wheel that omits the cube extension.

The existing suite passed all 302 tests with zero skips. Fourteen additional reproductions expose defects or unsupported numerical cases outside that suite. The fifteenth issue is missing source identity in the recorded experiment environment. Passing the existing tests does not close these findings.

[Repair checklist](REVIEW_CHECKLIST.md), [measured reproductions](history/results/review-findings.json), and [verification summary](history/results/review-summary.json) keep the review inspectable.

## Scope and execution evidence

Three independent reviewers examined the numerical core, neural and research code, and cube and dashboard code. The main review examined serialization, packaging, acceptance, setup, and end-to-end commands. Together the review covered every Python module and script, browser code, tests, and research documentation. This was code inspection plus executed checks. It was not an exhaustive proof of every possible input.

The host used Python 3.14.6, PyTorch 2.14.0+cu126, NumPy 2.5.2, and an NVIDIA RTX 4060 Laptop GPU. The full existing test run had CUDA available. The new smoke experiments and defect reproductions ran on CPU. Timing from these short runs measures this host and workload.

| Check | Current evidence | Boundary |
| --- | --- | --- |
| Existing test suite | 302 passed, zero failures, zero errors, zero skips, 19.36 seconds | Existing contracts and selected CPU/CUDA cases |
| Scalar and research CPU smoke | 17 records, 92 precision rows, seed 7, four distributions with 4,096 values each | Short execution check, not the million-value pilot or convergence evidence |
| Real dataset CPU smoke | 11 records, seed 7, official MNIST and CIFAR-10, actual Tiny Shakespeare | 256 train and 128 official test images per image task, 128 train and 64 validation language windows, one epoch |
| Standalone train command | FP32 seed 31, scalar smoke, two records including the integer inference control | Exercises the train wrapper; the main smoke also exercises chroma8 |
| New wheel build and isolated package import | Cube inverse exact, Fourier decode finite, browser assets present | Separate installation target and isolated Python import, with explicitly reused host dependencies |
| Committed wheel inspection | All six cube implementation and browser files absent | The package stored in `artifacts/packages/` is stale |
| Browser check | Real Cube Lab created 81 cells, turn and undo restored displayed rows exactly, 50% spectral request retained 40 of 81 coefficients | In-app browser on the local server; reconstruction MSE displayed as 0.027552; no captured warnings or errors |
| Additional defect checks | Fourteen reproduced issues, including official MNIST count and in-memory acceptance negative check | See the diagnostic script and JSON for inputs, seeds, source hashes, and measured values |

The scalar smoke used 512 training and 1,024 validation examples. Its periodic task used the same counts. Attention used 512 training and 128 validation sequences. Neural RGB fitting used 1,024 training and 512 validation observations. These sources remain distinct from the real datasets. The JSON records include dataset hashes, precision, configuration, storage, and measured runtime.

The canonical pilot still contains 95 model records and 92 precision rows. Its model artifact references resolved during this review. The recorded cube experiment still contains 16 geometry cases, 80 erasure cases, and six models. Those recorded runs were inspected. This review did not rerun the full pilot or its 24-epoch cube suite. It reran two short cube fits in a temporary directory to reproduce the artifact overwrite issue.

The quantized language and attention metrics in both the canonical record and the new smoke records are affected by R001. They show execution, but their causal interpretation and comparison with FP32 require repair and rerunning. The exact chroma8/INT8 and RGB24/fixed24 equivalence checks survive because each pair shares the same behavior.

## Architecture assessment

| Component | What the code actually does | Assessment |
| --- | --- | --- |
| Scalar codecs | Packs RGB24 into three bytes, rearranges finite FP32 into RGBD bytes, quantizes hue, fits scalar Lloyd tables, trains neural RGB encoder and decoder | Real implementations with numerical edge and dtype defects listed below |
| ChromaTensor | Holds code storage, a codec, and output dtype; shape operations manipulate integer tensors | Clear storage boundary; Python boolean indexing violates PyTorch semantics |
| Tokenizer | Stores scalar IDs or active residual tokens plus lengths and codec configuration | Actual compact residual tokens; codec precision must survive configuration reconstruction |
| Operators | Provides decode/compute/encode, finite integer table gathering, and integer saturated grid addition | Honest execution descriptions; grid addition needs a safe result container |
| Autograd and neural layers | Uses floating point master parameters and dense kernels with straight-through fake quantization | Working training; activation ranges break causal execution and generic conversion has silent coverage gaps |
| Conversion | Copies supported layers and restores ordinary master parameters | Covered child-to-child ties work; parent-to-child ties break; ordinary MHA can bypass inserted quantizers |
| Tensor and model files | JSON, fixed codec classes, checked byte counts, SHA-256 tensor checksums, bounded decompression, non-pickle arrays | Useful corruption and allocation checks; learned configuration export loses non-default parameter dtypes |
| Cube | Uses integer rotations for codes and int64 persistent identities | Supports exact inverse and four-turn claims; identity bytes count separately from color bytes |
| Distributed field | Stores a full FFT spectrum as phase8 and magnitude16 plus a scale | Real floating point spectral representation with redundant conjugate coefficients; missing coefficients decode as zero |
| Cube reconstruction | Fits an FP32/chroma8 MLP on independent low-rank periodic fields, with a noise negative control | Narrow, legitimate learning experiment; artifact publication needs immutable run directories |
| Research runner | Uses seeds, paired data, held-out samples, loss curves, timing, storage, and operation estimates | More than a visual demo; full MNIST counts, source identity, and acceptance verification need repair |
| Dashboard | Reads recorded JSON and calls real Python codecs and cube operations; escapes rendered text and serves bounded requests | Working local inspection UI; hostname validation is missing |
| Distribution | Builds an installable Python wheel with bundled browser assets | Current source builds correctly; the committed wheel differs from the claimed extension |

The strongest design choice is the explicit separation between integer storage and floating point execution. The code does not retain a secret original tensor inside ChromaTensor or HolographicCube. The lookup path also has tests that prohibit scalar decoding during its runtime calls. This makes the representation claims testable.

The main architectural weakness is verification. Mutable checkpoint names, absent source identity, and permissive acceptance conditions let a green record outlive the exact code or package it supposedly verifies. Model conversion also needs an explicit coverage report that identifies executed quantizers and unsupported modules. Replacing a child class alone is insufficient.

## Findings

P1 findings block acceptance of a central claim or a distributed artifact. P2 findings require correctness, evidence, or boundary repairs. P3 findings cover less common numerical configurations.

### R001. P1. Future tokens change earlier causal outputs

`chromatensor/autograd/quantization.py:137` computes activation minima and maxima across the entire tensor. Linear and embedding quantization use those ranges before the causal masks in `chromatensor/nn/attention.py:94` and `experiments/models.py:38`.

Changing a later token can therefore change the rounding of earlier activations. With seed 7, the actual language model receives `[1,2,3,4,5,6]` and `[1,2,3,10,20,30]`. The first three chroma8 logits change by a maximum of `0.029961705207824707`. INT8 gives the same change. FP32 gives zero. RGB24 and fixed24 each change by `5.736947059631348e-07`.

A direct ChromaAttention check with seed 19 changes only the final three positions. Earlier causal outputs change by `0.22101426124572754`.

Use fixed calibrated ranges or ranges that cannot inspect future positions. Cover every activation quantizer, including embeddings and normalization paths. Add prefix-invariance tests for the actual Transformer and both attention modes. Rerun affected language and attention experiments before accepting their causal metrics.

### R002. P1. Acceptance succeeds when every supplied test is skipped

`scripts/verify.py:19` counts tests and failures but ignores skips. Other gates accept finite losses, row counts, a fixed24 label, profiler list length, saved verification flags, or file existence. None ties the complete result to the currently distributed package.

The saved reproducer replaces only the XML input in memory with 302 tests and 302 skips. It leaves the actual result and artifact reads intact and intercepts the output write. `verify()` returns zero and reports `complete=true`. The canonical acceptance file remains untouched.

The independent reviewer also passed a fully negative in-memory fixture with zero training epochs and empty paired evidence. That demonstrates permissive gates, not fabrication of the existing results.

Require executed required tests, expected nonempty seed/task/control coverage, actual paired record hashes, measured samples, checkpoint reload checks, and source and package identity. Add negative acceptance fixtures that must fail.

### R003. P1. The committed wheel omits the cube extension

`artifacts/packages/chromatensor-0.1.0-py3-none-any.whl` has SHA-256 `6d87c386dda91b8c98aa2f5cb2fddb9e79280f09e808c03d520a675f5b70519c`. It contains none of `chromatensor/cube.py`, `chromatensor/holographic.py`, `experiments/cube.py`, `dashboard/cube_api.py`, `dashboard/cube.js`, or `dashboard/cube.css`. Its package exports and dashboard server also precede the extension.

The package verification record describes a separate installation and matching source hashes. It does not bind that evidence to the committed wheel. The new review build contains all six files and passes isolated import, inverse-turn, Fourier decode, and asset checks. Its SHA-256 is `67fabaefb10fb6c91e434214159ce8be8c1a0a74cbde29e77ddba37d4283c7fd`.

Replace or retire the stale distributed artifact, assign an appropriate version, and verify the exact wheel by hash. Keep this item open until the published artifact and its verification record agree.

### R004. P2. Conversion breaks parent-to-child parameter sharing

`chromatensor/nn/layers.py:277` traverses unchanged modules without registering their direct parameters in the alias map. The alias map only covers parameters of replaced layers. Restoration has the same omission at line 318.

A parent parameter shared with a child Linear weight starts as one trainable parameter. Conversion creates two distinct parameters. Restoring to Torch retains the split. This changes optimizer behavior even when fake quantization is disabled.

Preserve aliases across all modules and direct parameters. Test parent-to-child ties, ties across supported and unsupported modules, restoration, and optimizer parameter counts.

### R005. P2. Ordinary multi-head attention bypasses inserted quantizers

`chromatensor/nn/layers.py:264` replaces the Linear subclass used for `nn.MultiheadAttention.out_proj`. Ordinary MHA reads that child's projection parameters directly instead of calling its forward method.

Converting an eight-dimensional, two-head MHA at one bit inserts a ChromaLinear but executes zero quantizer calls. Its output matches the original exactly. The explicit `ChromaAttention.from_torch` adapter is a separate supported path.

Route supported attention through its adapter or reject generic conversion of that module. Report inactive quantizers and test conversion of real Transformer structures.

### R006. P2. Full-profile MNIST drops official observations

`experiments/datasets.py:19` assigns an equal quota per class, then truncates classes with fewer observations without redistributing the shortfall.

On cached official MNIST with seed 7, requesting the full 60,000 training images returns 58,862. Requesting the full 10,000 official test images returns 9,786. The 12,000/2,000 pilot counts are unaffected.

Select every observation when count reaches dataset size. Redistribute unmet quotas for smaller subsets. Assert exact counts and full coverage before describing the full profile as complete official splits.

### R007. P2. Learned codec export loses parameter precision

`chromatensor/codecs/learned.py:96` exports numerical values without a dtype. Reconstruction at line 104 uses the constructor's default dtype. `chromatensor/codecs/neural_rgb.py:140` and line 152 do the same for neural weights.

An FP64 four-centroid table returns as FP32 and changes decoded values by up to `9.934440770020814e-09` in the saved reproduction. An independent FP64 neural RGB check also changed output. Token and tensor file loading reconstruct these configurations, so recording a float64 output dtype does not preserve the lost decoder precision.

Record and restore parameter dtypes, or reject unsupported exports explicitly. Test configuration, token, and file roundtrips with FP64 parameters and a changed global default dtype.

### R008. P2. Python boolean indexing returns wrong values

`chromatensor/tensor.py:122` treats a Python boolean scalar as consuming a logical axis. Torch treats it as adding an axis without consuming one.

For `x = torch.arange(6.).reshape(2,3)`, `x[True,...,0]` returns `[[0,3]]`. The ChromaTensor equivalent returns `[[0,1,2]]`. Ellipsis expansion silently selects the wrong logical axis. Explicit boolean slicing can also raise when Torch accepts it.

Handle Python booleans like zero-dimensional boolean tensors. Test true and false values, ellipsis, explicit slices, new axes, and advanced indexing against Torch values and shapes.

### R009. P2. Learned scalar domains can collapse outside their range

`chromatensor/codecs/learned.py:27` validates endpoints in FP64, then creates the centroid parameter using default FP32 at line 28.

The accepted domain `[1e-50,2e-50]` becomes sixteen zero centroids. Every decoded observation lies outside the declared positive range. The maximum error is `2e-50`. A narrow interval near one also collapses to one centroid. Large finite endpoints can produce a low-level constructor failure.

Validate the centroid dtype's representable range and distinct endpoints, or accept an explicit centroid dtype. Return a clear error for unsupported domains.

### R010. P2. Two-symbol residual stages do not refine precision

`chromatensor/codecs/hierarchical.py:25` accepts a vocabulary of two symbols. The residual radius at line 39 divides by `(K-1)**level`, which stays one for that setting. Each residual stage has the same coarse spacing.

For `[-1,0,1]`, one stage reconstructs `[-1,-1,1]`, while two stages reconstruct `[-1,0,0]`. Additional stages alternate. More storage can destroy an exact endpoint without improving resolution. The documented error bound is constant for this case, so the bound itself is not false.

Reject the two-symbol multistage setting or implement a binary refinement scheme. Test that supported refinement settings deliver their stated behavior.

### R011. P2. Interrupted cube reruns can invalidate complete records

`experiments/cube.py:127` names checkpoints only by representation and seed. `scripts/run_cube_experiments.py:25` reuses that directory and publishes the new JSON only after training every model at line 54.

A two-epoch fit followed by a one-epoch fit in a temporary directory returns the same checkpoint path with a different hash. The first record reports MSE `0.018244193945020606` and 178,501 checkpoint bytes. Its replacement reports MSE `0.02210375674088328` and 178,402 bytes. An interrupted real rerun can leave the previous complete JSON pointing to the replacement checkpoint.

Use an immutable directory per run, record artifact hashes, verify the new set, and then publish the completed record. Test interruption after the first replacement candidate while an earlier complete run exists.

### R012. P2. Local dashboard requests accept foreign hostnames

`dashboard/server.py:135` does not validate Host. Its origin check at line 172 compares the origin with that same untrusted header.

A server bound to `127.0.0.1` returns HTTP 200 for both results reads and real tensor computation with `Host: attacker.example` and `Origin: http://attacker.example`. This confirms missing server-side hostname protection. Browser DNS rebinding exploitation depends on browser and network restrictions and was not demonstrated.

Validate permitted binding hostnames and exact origin scheme, hostname, and port. Test foreign Host headers on reads and computation. Continue documenting the trusted local hosting scope.

### R013. P2. Recorded experiment environments omit source identity

`experiments/common.py:31` uses `CHROMATENSOR_REVISION` if supplied and otherwise records `"working tree"`. The canonical pilot contains that default. Package versions and dataset hashes cannot identify the precise experiment implementation.

Record commit, dirty status, relevant source hashes, and configuration automatically. Record artifact hashes and a reproducible dependency set. Older runs need their original source evidence or a new run. Assigning the current commit to an older run would create unsupported provenance.

### R014. P3. Grid addition can overflow an accepted signed container

`chromatensor/operators/core.py:88` casts the saturated result back to the first operand's storage dtype. An eight-bit uniform tensor with valid code 100 in an int8 container passes construction. Adding it to itself should yield code 200, but the cast wraps negative and result validation raises `ValueError`.

Choose the output container from the codec's full code range, or require canonical storage dtypes at construction. Test signed and unsigned accepted input containers.

### R015. P3. Logarithmic normalization can divide by zero on valid endpoints

`chromatensor/codecs/rgb.py:35` subtracts logarithms of endpoints that can round to the same FP64 value. The accepted finite interval from `1e20` to its next FP64 neighbor gives NaN normalization for both endpoints and silently produces zero RGB codes.

Validate transformed bounds or compute the narrow logarithmic ratio with a stable formula. Test adjacent endpoints and defined failure behavior.

## Claims that hold and limits that remain

Default RGBD preserves finite FP32 bits, including signed zero and subnormals. RGB24 is actual three-byte storage and matches fixed24's numerical levels. Learned scalar palettes use numerical centroid fitting with deterministic color labels. Neural RGB is an actual trained byte-valued encoder and decoder. Compact residual token sequences omit inactive slots and account for length bytes.

Cube turns operate on integer codes and identities. The distributed representation uses real FFT kernels, stores phase and magnitude, and measures reconstruction after erasure. The held-out cube experiment learns a specified periodic-field generator. It carries a random-noise negative control and does not establish arbitrary-data recovery.

The documents correctly state that neural training and most arithmetic remain floating point. Chroma8 has eight nominal quantization bits but creates three physical RGB bytes during the current forward path. Master parameters, inspection buffers, metadata, and checkpoints incur separate costs. A lossless FP32 master checkpoint is not a deployed packed INT8 model.

The observed equal-level controls support the negative result that color labels alone do not improve these quantizers. They do not establish an advantage in accuracy, inference speed, training memory, hardware arithmetic, or novel numerical methods. The current models and two-seed pilot are too small and short for broad performance claims. The causal defect further limits language and attention evidence.

## Remaining software release work

The supported minimum dependency versions and Python 3.11 have not been verified in this review. The repo has no committed CI workflow or dependency lock. The rebuilt package check reused existing dependencies. An empty-machine install remains unverified.

The library has good local corruption checks, but a production release still needs deliberate malformed-file and resource-budget coverage, a defined compatibility policy, and release artifact verification. A checksum detects corruption and does not authenticate an author. RGBD's documented FP32 checkpoint scope must remain clear for FP64 or complex models.

For public hosting, authentication, request concurrency limits, and deployment-specific controls remain outside the current trusted local dashboard scope. Human screen-reader testing and sustained large-tensor or full-training performance also remain unverified.

Fix R001, R002, and R003 first. Then repair conversion semantics, full dataset selection, decoder precision, and artifact identity. Close each checklist item only after its stated regression check passes and any affected experiment has fresh evidence.

## Reproduce the review

Run these commands from the source checkout. The review outputs have separate names from the canonical pilot.

```powershell
python -m pytest --junitxml=artifacts/review/tests.xml
python scripts/run_benchmarks.py --profile smoke --device cpu --seeds 7 --tasks scalar periodic precision operators attention --output results/review-smoke.json
python scripts/run_benchmarks.py --profile smoke --device cpu --seeds 7 --tasks mnist cifar language --representations fp32 int8 chroma8 --output results/review-data-smoke.json
python scripts/review_reproductions.py
```

The reproduction script uses temporary files for cube overwrite checks and replaces acceptance inputs only in memory. It emits observations rather than repairing code. The MNIST count check requires the cached official dataset and records an unavailable check if that cache is absent. It records source hashes and the diagnostic script hash. After repairs, the corresponding reproduction flags should turn false and new regression tests should establish the intended behavior.

Cube Lab after exact turn restoration and a measured partial reconstruction. Original artifact `artifacts/review/cube-browser.jpg` is omitted; regenerate it from source.
