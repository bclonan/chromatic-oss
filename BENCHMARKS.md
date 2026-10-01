# Benchmark protocol

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](docs/RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

`python scripts/run_benchmarks.py` writes a result after each completed model. It synchronizes CUDA around measured GPU timings. Main task training time counts optimization batches and excludes epoch validation, checkpoint export, and reconstruction diagnostics. The jointly learned RGB codec's fitting timer includes the progress-loss evaluations inside its fit loop.

Main task inference uses two warm-up batches and ten timed batches. The attention ablation uses one warm-up and five timed batches. Neural RGB encoding and decoding each use two warm-ups and five timed calls. Cube model inference uses two warm-ups and five timed batches. Precision encoding and decoding are separate CPU timings over complete generated arrays. These scopes must remain attached to their records when comparing timings.

Process memory is RSS sampled every 10 milliseconds. It includes the Python interpreter, loaded data, imported libraries, and caches. The report also includes the change from starting RSS and the CUDA allocator peak where available. These are different measures. They do not give the full physical GPU memory use.

`bytes_model` in a trainable-model record is a diagnostic quantized payload plus serialized codec metadata estimate. `live_parameter_bytes` measures the actual master parameters. In the current FP32-master experiments, `checkpoint_bytes` measures a real lossless RGBD checkpoint containing that state. Default checkpoint export also supports lossless FP64 through numeric NPY entries. An explicitly supplied codec selects its own precision rules. [File formats](docs/FILE_FORMATS.md) records that policy. These quantities must not be substituted for each other. RGB24 uses three bytes per scalar. A 256-entry chromatic vocabulary can store one-byte IDs plus a palette, while its fake-quantization forward materializes three-byte RGB symbols. The execution reports identify those runtime buffers and their FP64 range metadata.

The storage experiment measures `.ctensor` files with and without zlib, FP32/FP16 `.npy` and compressed `.npz`, PyTorch `.pt`, and a truly packed INT4 file. Full file sizes include format overhead unless a row says payload only. Dense hierarchical tensors retain inactive token slots. Active tokens plus their lengths are an alternative serialization estimate, not the dense allocation.

The CPU integer model is a transparent reference implementation. It stores INT8 weights, multiplies INT32 operands, and restores scale and bias in FP32. It is not an optimized INT8 kernel and cannot be used to infer commercial backend speed.

Lookup tables use at most 1,024 entries per operand. A reduced-vocabulary addition experiment compares full decode/compute/encode, integer table lookup, integer grid addition, and a learned MLP mapping. The optimization report compares cached table gather with the decoded reference and records that it excludes the output tensor wrapper. The operator table reports the complete wrapper costs separately.

Reproduce the pilot with `--seeds 7 23 --profile pilot` and an explicit device. The current canonical pilot uses CUDA; its two checkpoint-backed profiler forwards use CPU. Use `--resume` after interruption with the same arguments. Use `--profile full` to change the sample and epoch budgets. Wall times vary with background work and hardware. Seeds, data hashes, and deterministic-algorithm settings make repetitions inspectable. Floating-point results can still differ across hardware and dependency versions. Saved inference probes check those differences against recorded tolerances.

The pilot has two seeds. Paired mean differences and sample standard deviations describe those observations. They are not confidence bounds or proof of a general advantage. Repeated significance tests over this small matrix would not improve that evidence.
