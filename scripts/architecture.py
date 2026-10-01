"""Build and check the offline, source-backed architecture explorer.

The registry below contains reviewed semantic relationships. Import discovery is
reported separately and never promoted to a runtime or data-flow assertion.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("docs/architecture")
GROUPS = [
    {"id": "native", "name": "Native hue engine", "description": "Rust stores literal RGBD tokens and computes in Z/391680Z."},
    {"id": "entry", "name": "Native entry points", "description": "Python, C and the standalone CLI reach the same Rust engine."},
    {"id": "legacy", "name": "Legacy numerical APIs", "description": "PyTorch codecs, decoded operators and floating training remain explicit research APIs."},
    {"id": "dashboard", "name": "Local dashboard", "description": "A local HTTP server reads saved results and runs bounded legacy inspections."},
    {"id": "research", "name": "Experiments and evidence", "description": "Seeded datasets, measurements and source identities support research checks."},
    {"id": "release", "name": "Build and release", "description": "Offline Cargo builds, wheel assembly and acceptance checks produce local artifacts."},
]


def ref(path, symbol=None, contains=None):
    return {"path": path, **({"symbol": symbol} if symbol else {}), **({"contains": contains} if contains else {})}


def node(id, name, group, purpose, inputs, outputs, effects, ownership, trust, failures, sources, tests=(), kind="component"):
    return dict(id=id, name=name, group=group, purpose=purpose, inputs=inputs, outputs=outputs,
                side_effects=effects, ownership=ownership, trust_boundary=trust, failures=failures,
                sources=sources, tests=list(tests), kind=kind, dependencies=[])


NODES = [
    node("tokens", "RGBD token domain", "native", "Map canonical saturated RGB plus D to one of 391,680 integer phase ticks.",
         "RGB bytes, canonical RGBD bytes or bounded integer ticks", "Four-byte Token values or integer phases", "No I/O. General RGB projection discards saturation and brightness.",
         "Rust Token value, copied into tensor storage", "Unvalidated colors and phases enter a finite domain", "Gray RGB, noncanonical RGBD or out-of-domain ticks fail.",
         [ref("native/hue_tensor/src/lib.rs", "Token"), ref("native/hue_tensor/src/lib.rs", "Token::from_rgb")], [ref("tests/test_native_hue.py", "test_exact_color_corners_fine_channel_and_public_aliases")]),
    node("tensor", "Owned Rust tensor", "native", "Own Vec<Token> storage and a logical shape. Rank and element budgets bound allocations.",
         "Validated tokens and dimensions", "Owned tensor with copied token bytes", "Allocates memory. No hidden real-number tensor.", "Rust Tensor owns both vectors. Constructors create independent outputs.",
         "Shape/count validation precedes allocation", "Rank above 32, count above 67,108,864, overflow or mismatched shape fails.",
         [ref("native/hue_tensor/src/lib.rs", "Tensor"), ref("native/hue_tensor/src/lib.rs", "shape_len")], [ref("tests/test_native_hue.py", "test_invalid_shapes_operations_and_closed_handles_fail")]),
    node("operators", "Integer hue operators", "native", "Add, subtract and multiply modulo 391,680. Contract rank-two matrices, reduce axes and copy shape transforms.",
         "Owned operands, axes, ticks or byte blend weight", "New owned tensor", "Allocates output. Inputs remain unchanged.", "Rust reads operands and owns outputs", "Integer phase labels do not imply real arithmetic or physical color mixing", "Incompatible broadcast, contraction, slice or axis fails. Division, GPU and autograd are absent.",
         [ref("native/hue_tensor/src/lib.rs", "matmul"), ref("native/hue_tensor/src/lib.rs", "blend"), ref("native/hue_tensor/src/lib.rs", "slice")], [ref("tests/test_native_hue.py", "test_random_modular_algebra_and_immutability")]),
    node("storage", "Native file codec", "native", "Save literal RGBD with shape metadata and SHA-256. Load only after length, budget, checksum and color validation.",
         "Tensor or file path and byte budget", "HUECT file or new owned tensor", "Reads files. Writes a synced adjacent temporary file and renames it.", "Rust loader owns loaded buffers. Writer cleans temporary files on failure.",
         "Untrusted file bytes enter memory. SHA-256 checks corruption, not author identity.", "Oversize, malformed, trailing, corrupt or noncanonical files fail. Rename rules depend on the OS.",
         [ref("native/hue_tensor/src/storage.rs", "save_tensor"), ref("native/hue_tensor/src/storage.rs", "load_tensor")], [ref("tests/test_native_hue.py", "test_corrupt_or_adversarial_files_fail_even_after_rehashing")], "store"),
    node("ffi", "C ABI boundary", "entry", "Expose opaque handles, output capacities and status codes. Catch Rust panics at exported operations.",
         "Caller pointers, handles, shapes, buffers and UTF-8 paths", "Status 0, 1, 2 or 3, output handle, copied buffers and thread-local error", "Allocates and frees native handles. Storage calls access files.",
         "C caller must free once and keep handles alive for every call", "Declared capacities are checked. Arbitrary pointer validity cannot be proven.", "Invalid inputs reject. Dangling, aliased or concurrently freed pointers violate the C contract.",
         [ref("native/hue_tensor/src/ffi.rs", "boundary"), ref("native/hue_tensor/src/ffi.rs", "ht_free"), ref("native/hue_tensor/src/ffi.rs", "ht_load")], [ref("tests/test_native_hue.py", "test_parallel_close_read_and_reversed_operand_calls_do_not_crash_or_deadlock")]),
    node("python", "HueTensor Python binding", "entry", "Load a checkout library or the separately installed native wheel through ctypes. Acquire operand locks in object order and manage one opaque handle per instance.",
         "Python integer sequences, byte buffers and paths", "HueTensor objects and copied RGBD, RGB or phases", "Loads a local DLL or shared library, allocates handles and accesses native files.",
         "Instance owns a handle. close is idempotent. Context exit closes it.", "Python types and paths cross into C. Native ABI version and modulus must match.", "Missing or incompatible binary, NUL path, closed handle, invalid dtype or native rejection raises an exception.",
         [ref("chromatensor/hue.py", "_library_path"), ref("chromatensor/hue.py", "_library"), ref("chromatensor/hue.py", "_owned"), ref("chromatensor/hue.py", "HueTensor")], [ref("tests/test_native_hue.py", "test_NUL_path_is_rejected_before_native_IO_and_does_not_overwrite_target")]),
    node("package", "Public package names", "entry", "Export HueTensor, Tensor and ColorTensor. Resolve legacy names only when a caller requests them.",
         "Python imports and requested attribute name", "Native aliases or explicit legacy API", "Caches resolved legacy attributes in the module.", "Python import machinery owns module state", "Choosing a legacy name loads PyTorch research code", "Unknown names raise AttributeError. Native execution is unavailable until a library is present.",
         [ref("chromatensor/__init__.py", "__getattr__")], [ref("tests/test_native_hue.py", "test_exact_color_corners_fine_channel_and_public_aliases")]),
    node("cli", "Standalone Rust CLI", "entry", "Parse encode, decode, save, read, arithmetic and demo commands without Python.",
         "Command arguments and local native files", "JSON output and native tensor files", "Reads and writes requested files. Prints output and errors.", "CLI owns tensors until process exit", "Untrusted text and files cross into bounded Rust constructors and storage", "Bad arity, invalid numbers, dimensions, file limits or operations exit with an error.",
         [ref("native/hue_tensor/src/bin/hue_tensor.rs", "run"), ref("native/hue_tensor/src/bin/hue_tensor.rs", "demo")], [ref("scripts/verify_native_hue_release.py", "validate_cli_demo")]),
    node("header", "C caller contract", "entry", "Declare the exported ABI and its ownership, buffer and thread rules for C consumers.",
         "Consumer compile and API calls", "Opaque handle declarations and function signatures", "No runtime work in the header itself.", "External C consumer owns its buffers and handle lifetimes", "Header documents obligations that a raw pointer ABI cannot enforce", "Misdeclared buffers, double free and concurrent free can cause undefined behavior.",
         [ref("native/hue_tensor/include/hue_tensor.h", contains="typedef struct ht_tensor")], [ref("scripts/verify_native_hue_release.py", "validate_native_commands")]),
    node("legacy-tensor", "Legacy ChromaTensor", "legacy", "Store encoded PyTorch color tensors and decode real numerical values through a scalar codec.",
         "PyTorch tensors, code tensors and ScalarCodec", "Encoded tensor or floating decoded tensor", "Allocates PyTorch tensors. May use the selected device.", "Python object owns PyTorch codes and codec state", "Legacy RGBD uses D as an IEEE exponent. It is incompatible with native D sub-hue bytes.", "Invalid code shape, dtype or codec configuration fails.",
         [ref("chromatensor/tensor.py", "ChromaTensor")], [ref("tests/test_tensor.py")]),
    node("codecs", "Scalar codecs and vocabulary", "legacy", "Encode scalar values with RGB, RGBD, HSV, uniform or learned codebooks. Tokenizers preserve vocabulary metadata.",
         "Real scalar tensors, quantization parameters or learned samples", "Color codes, token IDs and decoded floats", "Fitting changes codebooks and allocates tensors.", "Codec or vocabulary owns its configuration and learned state", "Finite values, code ranges and metadata govern decode semantics", "Invalid numerical values, codes, vocabulary IDs or parameter ranges fail.",
         [ref("chromatensor/codecs/base.py", "ScalarCodec"), ref("chromatensor/codecs/rgbd.py", "RGBDCodec"), ref("chromatensor/tokenizer/__init__.py", "TokenSequence")], [ref("tests/test_codecs.py"), ref("tests/test_rgbd_tokens.py")]),
    node("legacy-ops", "Decoded PyTorch operators", "legacy", "Decode operands, run PyTorch arithmetic and encode the result.",
         "Legacy ChromaTensor operands", "Legacy encoded output or floating similarity", "Allocates decoded intermediates and output codes.", "PyTorch manages buffers", "These operations execute floating point and do not call the native ring operators", "Wrong operand types and invalid PyTorch shapes fail.",
         [ref("chromatensor/operators/core.py", "_decoded")], [ref("tests/test_operators.py")]),
    node("nn", "Quantized neural layers", "legacy", "Quantize and reconstruct weights or activations around PyTorch layers with straight-through gradients.",
         "Floating masters, activations, quantization configuration", "Floating predictions, gradients and encoded state reports", "Training updates floating parameters. Device kernels run through PyTorch.",
         "Modules own parameters and quantizer state", "Quantization-aware training does not establish integer or entirely chromatic execution", "Unsupported representation, invalid bit configuration or layer shape fails.",
         [ref("chromatensor/nn/layers.py", "ChromaLinear"), ref("chromatensor/autograd/quantization.py", "_StraightThrough"), ref("chromatensor/autograd/quantization.py", "ChromaQuantizer")], [ref("tests/test_nn.py"), ref("tests/test_rgbd_nn.py")]),
    node("cubes", "Cube research", "legacy", "Rearrange encoded cells in ChromaCube and reconstruct Fourier fields in HolographicCube.",
         "Encoded cell grid, twists, masks or real fields", "Rearranged cube, reconstructed field and metrics", "Allocates tensors and FFT intermediates.", "Cube owns codes, identities, configuration and metadata", "Fourier computation uses floating and complex transforms. Fragment recovery depends on the declared experiment.",
         "Invalid geometry, masks, codes, finite scale or codec fails.", [ref("chromatensor/cube.py", "ChromaCube"), ref("chromatensor/holographic.py", "HolographicCube")], [ref("tests/test_cube.py"), ref("tests/test_holographic.py")]),
    node("legacy-storage", "Legacy files and checkpoints", "legacy", "Serialize bounded color tensor files and model checkpoints. Decode archives without arbitrary pickle execution.",
         "Legacy tensor, model state or archive and budgets", "CTENSOR files, checkpoint archives and loaded legacy state", "Reads, decompresses and atomically writes local files.", "Loader owns loaded tensors and metadata", "Archive entries, expanded lengths, JSON and NumPy payloads cross validation boundaries", "Traversal, oversized members, inconsistent shapes, corrupt checksums or incompatible metadata fail.",
         [ref("chromatensor/serialization/format.py", "_open_archive"), ref("chromatensor/serialization/format.py", "load_chroma_checkpoint")], [ref("tests/test_serialization.py"), ref("tests/test_file_limits.py")], "store"),
    node("browser", "Dashboard browser UI", "dashboard", "Display recorded metrics, colors and cubes. Submit bounded inspection inputs to the local server.",
         "Saved result JSON and inspector responses", "Charts, tables and downloadable JSON", "Fetches local API endpoints. Export creates browser Blob downloads.", "Browser memory owns UI and cube state", "Browser inputs are untrusted to the local server. Recorded metrics remain distinct from live inspection.", "Missing results or rejected inspection appears as an error. Browser does not run native hue operators.",
         [ref("dashboard/app.js", contains="async function encodeInput"), ref("dashboard/cube.js", contains='fetch("/api/cube"')], [ref("tests/test_dashboard.py", "test_http_serves_measured_results_and_static_assets")]),
    node("http", "Local HTTP boundary", "dashboard", "Serve allowlisted static files, recorded results and bounded POST inspections. Validate exact local Host and Origin authorities.",
         "Local HTTP requests, JSON bodies and artifact paths", "HTTP results or explicit error status", "Reads results and permitted artifacts. Runs codec inspections on POST.", "ThreadingHTTPServer owns connections and handlers", "Host, Origin, body size and resolved artifact paths gate local access. There is no account authentication.",
         "Foreign Host returns 421, cross-origin POST 403, wrong type 415, oversized body 413. Missing records return 404.",
         [ref("dashboard/server.py", "DashboardHandler"), ref("dashboard/server.py", "DashboardServer")], [ref("tests/test_dashboard.py", "test_foreign_host_cannot_read_or_compute_even_with_matching_origin")]),
    node("inspection", "Legacy tensor and cube inspector", "dashboard", "Validate finite input values and cube state, then call the actual legacy codecs and cube APIs.",
         "Values, shapes, codec name or cube operation and state", "Codes, decoded values, geometry and measured error", "Allocates bounded PyTorch state. It can read selected saved samples.", "Request creates temporary Python tensors. Cube state returns to the browser.", "Request JSON crosses type, shape and allocation limits", "Invalid values, shape, codec, cell identity or geometry returns a rejection. Missing PyTorch returns 503.",
         [ref("dashboard/server.py", "inspect_tensor"), ref("dashboard/cube_api.py", "inspect_cube"), ref("dashboard/cube_api.py", "_restore")], [ref("tests/test_dashboard.py", "test_inspector_rejects_invalid_or_unbounded_requests"), ref("tests/test_cube_api.py")]),
    node("datasets", "Dataset loaders", "research", "Select seeded official image subsets, sampled language windows and separately labeled analytic scalar data.",
         "Task, seed, counts, cache path and device", "Train and validation tensors with source URLs and hashes", "May download MNIST, CIFAR-10 and Tiny Shakespeare into the requested cache.", "Experiment runner owns downloaded cache and selected tensors", "External dataset bytes enter training. Hashes record identity, not a fixed upstream authenticity guarantee.",
         "Download, decode, missing cache or invalid sample count fails. No synthetic substitution under a real dataset name.",
         [ref("experiments/datasets.py", "image_data"), ref("experiments/datasets.py", "language_data"), ref("experiments/datasets.py", "scalar_data")], [ref("tests/test_datasets.py")], "integration"),
    node("training", "Research runners", "research", "Run seeded training, attention, ablations and cube comparisons with explicit precision and equal-budget controls.",
         "Configuration, datasets, seeds, representation and device", "Measurements, curves, samples and model checkpoints", "Trains models and writes artifacts and JSON reports.", "Runner owns temporary models and run directory", "Execution metadata must distinguish floating training, integer controls and native operators", "Unavailable device, data, dependency or failed experiment must remain visible in evidence.",
         [ref("experiments/training.py", "train_one"), ref("scripts/run_benchmarks.py", "main"), ref("scripts/run_cube_experiments.py", "main")], [ref("tests/test_training.py"), ref("tests/test_cube_experiments.py")]),
    node("provenance", "Provenance and result checks", "research", "Bind configuration, executable source and artifact hashes to measured records. Check coverage, precision and artifact paths.",
         "Run records, source files, config and artifact paths", "Verified identities or gate failures", "Reads and hashes source, artifacts and metadata.", "Report owns a manifest snapshot. Checker compares it to current files.", "Recorded claims cross into acceptance only after independent checks", "Source drift, missing artifacts, unsafe paths, false coverage or execution claims fail.",
         [ref("experiments/provenance.py", "source_manifest"), ref("experiments/provenance.py", "verify_provenance"), ref("experiments/verification.py", "verify_precision")], [ref("tests/test_acceptance.py"), ref("tests/test_release_research.py")]),
    node("results", "Local results and artifacts", "research", "Store measured JSON, saved models, native tensor files and receipts in results and artifacts directories.",
         "Runner output, checkpoints and evidence", "Files read by checkers and dashboard", "Persistent local writes. No cloud database or remote synchronization.", "Workspace owner controls local files", "Results are mutable local files. Checksums alone do not authenticate the author.", "Absent, stale, corrupt or incompatible records cannot establish current acceptance.",
         [ref("experiments/provenance.py", "artifact_manifest"), ref("dashboard/server.py", contains='if request.path in ("/api/results", "/api/cube-results")')], [ref("tests/test_dashboard.py", "test_invalid_results_cannot_be_presented_as_measurements")], "store"),
    node("reporting", "Research report writer", "research", "Compute paired comparisons and write conclusions based on recorded measurements.",
         "Run records and comparison pairs", "RESULTS prose and comparison metadata", "Writes report documents.", "Run records remain the measurement source", "A report does not add hardware, energy, novelty or generalization evidence", "Missing or unsupported metrics limit conclusions. Native CPU results do not establish a speed benefit.",
         [ref("experiments/reporting.py", "paired_comparisons"), ref("experiments/reporting.py", "conclusions")], [ref("tests/test_release_research.py")]),
    node("build", "Native builder and wheel", "release", "Check Rust attribution hashes and compiler identity, build in offline mode and assemble a wheel containing the exact binding, C header and runtime notices.",
         "Rust sources, binding, attribution manifest, matching compiler and target platform", "Library, CLI, platform wheel and hashes", "Runs Cargo. Linux cross-build runs a pinned container. Writes platform build, native library and wheel directories.",
         "Builder owns build outputs. Installer owns installed files.", "Build toolchain and platform binary cross the packaging boundary", "Missing Cargo, container, unsupported target or build failure aborts. Build output does not prove runtime correctness.",
         [ref("scripts/build_native_hue.py", "rust_notices"), ref("scripts/build_native_hue.py", "make_wheel"), ref("scripts/build_native_hue.py", "main")], [ref("tests/test_native_release.py")]),
    node("acceptance", "Native release gates", "release", "Check fresh sources, host tests, Rust builds, independent integer oracle, CLI proof and isolated wheel environments.",
         "Current source, binaries, receipts and installed-wheel test records", "Native acceptance report with passed or failed gates", "Reads artifacts and writes verification JSON. Package runner creates isolated environments.",
         "Checker owns the report. Each receipt identifies its execution layer.", "Historical evidence cannot claim acceptance for changed sources or binaries", "Source mismatch, tampered oracle, missing operation, invalid package matrix or failed command fails acceptance.",
         [ref("scripts/verify_native_hue_release.py", "verify"), ref("scripts/verify_native_hue.py", "main"), ref("scripts/run_native_hue_package_checks.py", "main")], [ref("tests/test_native_release.py"), ref("tests/test_native_hue.py", "test_rehashed_integer_control_tampering_and_false_execution_claims_are_rejected")]),
    node("legacy-gates", "Legacy research gates", "release", "Select the separate legacy acceptance path and check research coverage, artifacts and dependency matrix.",
         "Legacy result and test records", "Separate legacy acceptance JSON", "Reads records and writes a report.", "Legacy checker owns a report separate from native release", "Passing research gates does not establish native GPU, autograd or neural training", "Missing tests, stale source, incomplete matrix or inconsistent metrics fails.",
         [ref("scripts/verify.py", "verify")], [ref("tests/test_acceptance.py"), ref("tests/test_release_matrix.py")]),
]


def edge(id, source, target, type, label, evidence, boundary=False, flows=()):
    return dict(id=id, source=source, target=target, type=type, label=label,
                evidence=evidence, boundary=boundary, flows=list(flows))


EDGES = [
    edge("package-python", "package", "python", "call", "Native imports and aliases", [ref("chromatensor/__init__.py", contains="from .hue import")], flows=["native"]),
    edge("python-ffi", "python", "ffi", "call", "ctypes with ordered handle locks", [ref("chromatensor/hue.py", "_owned"), ref("chromatensor/hue.py", "_library")], True, ["native", "save-load"]),
    edge("header-ffi", "header", "ffi", "contract", "Declares C ABI and caller obligations", [ref("native/hue_tensor/include/hue_tensor.h", contains="void ht_free")], True),
    edge("ffi-tensor", "ffi", "tensor", "ownership", "Allocate opaque handle, free once", [ref("native/hue_tensor/src/ffi.rs", "ht_new_rgbd"), ref("native/hue_tensor/src/ffi.rs", "ht_free")], True, ["native"]),
    edge("tokens-tensor", "tokens", "tensor", "data", "Validated literal RGBD tokens", [ref("native/hue_tensor/src/lib.rs", "from_rgbd")], flows=["native"]),
    edge("tensor-operators", "tensor", "operators", "data", "Borrow token and shape inputs", [ref("native/hue_tensor/src/lib.rs", "matmul")], flows=["native"]),
    edge("operators-tokens", "operators", "tokens", "call", "Encode integer results into tokens", [ref("native/hue_tensor/src/lib.rs", "Tensor::binary")], flows=["native"]),
    edge("operators-tensor", "operators", "tensor", "ownership", "Return a new owned output", [ref("native/hue_tensor/src/lib.rs", "reshape")], flows=["native"]),
    edge("ffi-storage", "ffi", "storage", "call", "ht_save and ht_load", [ref("native/hue_tensor/src/ffi.rs", "ht_save"), ref("native/hue_tensor/src/ffi.rs", "ht_load")], True, ["save-load"]),
    edge("storage-tensor", "storage", "tensor", "read", "Read and validate saved RGBD and shape", [ref("native/hue_tensor/src/storage.rs", "load_tensor")], True, ["save-load"]),
    edge("tensor-storage", "tensor", "storage", "write", "Write literal tokens, shape and checksum", [ref("native/hue_tensor/src/storage.rs", "save_tensor")], True, ["save-load"]),
    edge("cli-tensor", "cli", "tensor", "call", "Direct Rust constructors", [ref("native/hue_tensor/src/bin/hue_tensor.rs", "run")], True, ["cli"]),
    edge("cli-operators", "cli", "operators", "call", "Direct Rust arithmetic", [ref("native/hue_tensor/src/bin/hue_tensor.rs", "run")], flows=["cli"]),
    edge("cli-storage", "cli", "storage", "call", "Bounded file read and save", [ref("native/hue_tensor/src/bin/hue_tensor.rs", "load"), ref("native/hue_tensor/src/bin/hue_tensor.rs", "write_and_print")], True, ["cli"]),
    edge("package-legacy", "package", "legacy-tensor", "call", "Requested legacy name loads its module", [ref("chromatensor/__init__.py", "__getattr__")], True),
    edge("legacy-codecs", "legacy-tensor", "codecs", "call", "Encode and decode scalar values", [ref("chromatensor/tensor.py", "ChromaTensor")]),
    edge("legacy-ops-tensor", "legacy-ops", "legacy-tensor", "call", "Decode, PyTorch operation, re-encode", [ref("chromatensor/operators/core.py", "_decoded")], flows=["training"]),
    edge("nn-codecs", "nn", "codecs", "data", "Quantizer encoded state and color codes", [ref("chromatensor/autograd/quantization.py", "ChromaQuantizer")], flows=["training"]),
    edge("cubes-legacy", "cubes", "legacy-tensor", "call", "Encoded ChromaTensor cell grid", [ref("chromatensor/cube.py", "ChromaCube")], flows=["inspection"]),
    edge("legacy-storage-tensor", "legacy-storage", "legacy-tensor", "data", "Legacy codes and codec configuration", [ref("chromatensor/serialization/format.py", "load_ctensor")]),
    edge("browser-http", "browser", "http", "http", "Local GET results, POST inspections", [ref("dashboard/app.js", contains='fetch("/api/results"'), ref("dashboard/cube.js", contains='fetch("/api/cube"')], True, ["inspection"]),
    edge("http-browser", "http", "browser", "http", "JSON responses and static assets", [ref("dashboard/server.py", "DashboardHandler")], True, ["inspection"]),
    edge("http-inspection", "http", "inspection", "call", "Validated JSON request", [ref("dashboard/server.py", contains="result = inspect_tensor(payload)")], True, ["inspection"]),
    edge("inspection-codecs", "inspection", "codecs", "call", "Actual legacy codec encode and decode", [ref("dashboard/server.py", "inspect_tensor")], flows=["inspection"]),
    edge("inspection-cubes", "inspection", "cubes", "call", "Restore, twist, inspect and reconstruct", [ref("dashboard/cube_api.py", "inspect_cube")], flows=["inspection"]),
    edge("results-http", "results", "http", "read", "Read allowlisted local results and artifacts", [ref("dashboard/server.py", contains='if request.path == "/api/artifact"')], True, ["inspection"]),
    edge("datasets-training", "datasets", "training", "data", "Seeded splits, counts and dataset hashes", [ref("experiments/training.py", contains="from .datasets import")], True, ["training"]),
    edge("training-nn", "training", "nn", "call", "Floating training and quantized layers", [ref("experiments/training.py", "train_one")], flows=["training"]),
    edge("training-cubes", "training", "cubes", "call", "Cube geometry and learning experiments", [ref("experiments/cube.py", contains="from chromatensor")]),
    edge("training-storage", "training", "legacy-storage", "call", "Measured checkpoint storage", [ref("experiments/training.py", "storage_metrics")], True, ["training"]),
    edge("training-results", "training", "results", "write", "Write measurements and saved artifacts", [ref("scripts/run_benchmarks.py", "main")], True, ["training"]),
    edge("results-provenance", "results", "provenance", "read", "Read, hash and validate saved run evidence", [ref("experiments/provenance.py", "verify_provenance")], True, ["release"]),
    edge("results-reporting", "results", "reporting", "data", "Supply paired measured records", [ref("experiments/reporting.py", "conclusions")]),
    edge("python-build", "python", "build", "data", "Exact binding copied to the native wheel", [ref("scripts/build_native_hue.py", "make_wheel")], True, ["release"]),
    edge("header-build", "header", "build", "data", "C header copied to wheel", [ref("scripts/build_native_hue.py", "make_wheel")], flows=["release"]),
    edge("build-acceptance", "build", "acceptance", "data", "Built binaries, wheel hashes and receipts", [ref("scripts/verify_native_hue_release.py", "verify")], True, ["release"]),
    edge("provenance-acceptance", "provenance", "acceptance", "call", "Current source and artifact identities", [ref("scripts/verify_native_hue_release.py", contains="from experiments.provenance import")], flows=["release"]),
    edge("acceptance-results", "acceptance", "results", "write", "Write native acceptance report", [ref("scripts/verify_native_hue_release.py", "verify")], True, ["release"]),
    edge("provenance-legacy", "provenance", "legacy-gates", "call", "Legacy source and research evidence", [ref("scripts/verify.py", "verify")]),
    edge("legacy-results", "legacy-gates", "results", "write", "Write separate legacy acceptance report", [ref("scripts/verify.py", "main")], True),
]

FLOWS = [
    {"id": "native", "name": "Python hue operation", "description": "Import a native alias, create RGBD storage through the ABI, borrow operands and return a new Rust tensor.", "steps": ["package", "python", "ffi", "tokens", "tensor", "operators"]},
    {"id": "save-load", "name": "Save and load native bytes", "description": "The binding passes a path and budget to Rust storage. Loading checks the file before allocating token storage.", "steps": ["python", "ffi", "storage", "tensor"]},
    {"id": "cli", "name": "Standalone CLI", "description": "Command text reaches the Rust constructors, operators and file codec directly.", "steps": ["cli", "tensor", "operators", "storage"]},
    {"id": "inspection", "name": "Inspect a legacy cube", "description": "The browser submits local JSON. The server validates authority and bounds, runs legacy codecs or cube operations and returns measured results.", "steps": ["browser", "http", "inspection", "codecs", "cubes", "results"]},
    {"id": "training", "name": "Run legacy training", "description": "Seeded dataset selection feeds floating training and quantization-aware layers. Checkpoints and measured results retain their provenance.", "steps": ["datasets", "training", "nn", "legacy-storage", "results"]},
    {"id": "release", "name": "Check a native release", "description": "Exact binding and header enter the wheel. The checker compares binaries, sources, integer oracle and isolated installation records.", "steps": ["python", "header", "build", "provenance", "acceptance", "results"]},
]

INTEGRATIONS = [
    {"name": "Native Rust dependencies", "status": "present", "detail": "Standard library only. Cargo.toml declares no third-party dependencies.", "sources": [ref("native/hue_tensor/Cargo.toml")]},
    {"name": "Native Python dependencies", "status": "present", "detail": "The standalone chromatic-hue wheel uses ctypes and other standard-library modules. The full legacy package has separate dependencies.", "sources": [ref("chromatensor/hue.py"), ref("scripts/build_native_hue.py", "make_wheel")]},
    {"name": "PyTorch and NumPy", "status": "present", "detail": "Legacy numerical APIs and research use these libraries. They are not native ring execution.", "sources": [ref("pyproject.toml", contains="dependencies =")]},
    {"name": "Dataset downloads", "status": "present", "detail": "torchvision image downloads and Tiny Shakespeare URL retrieval may use the network. The architecture explorer itself performs no network requests.", "sources": [ref("experiments/datasets.py", "image_data"), ref("experiments/datasets.py", "language_data")]},
    {"name": "Cargo and container toolchain", "status": "present", "detail": "Cargo runs with --offline. The Linux cross-build uses a pinned Rust container, which must already be available or pulled by the container runtime.", "sources": [ref("scripts/build_native_hue.py", "main")]},
    *[{"name": name, "status": "absent", "detail": detail, "sources": [ref("pyproject.toml"), ref("dashboard/server.py", "DashboardHandler")]} for name, detail in [
        ("Accounts and authentication", "No account model, sign-in flow or authorization service was found in the reviewed runtime. Local Host and Origin checks are not account authentication."),
        ("Chat and messaging", "No chat backend, message store or messaging integration was found."),
        ("Payments and billing", "No payment processor, checkout or billing code was found. Licensing terms do not implement payment collection."),
        ("Convex or cloud database", "No Convex client, schema or hosted database integration was found. Persistence uses local files."),
        ("Hosted inference providers", "No OpenAI, Anthropic or other hosted inference provider call was found. Legacy model inference runs locally through PyTorch."),
    ]],
]

REQUIRED_NODE_FIELDS = {"id", "name", "group", "purpose", "inputs", "outputs", "side_effects", "ownership", "dependencies", "trust_boundary", "failures", "sources", "tests", "kind", "status"}
EDGE_TYPES = {"call", "contract", "ownership", "data", "http", "read", "write"}


def resolve_reference(root, reference):
    path = Path(reference["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Source reference escapes repository: {path}")
    location = root / path
    text = location.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    symbol = reference.get("symbol")
    if symbol and path.suffix == ".py":
        candidates = [item for item in ast.walk(ast.parse(text))
                      if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == symbol]
        if len(candidates) != 1:
            raise ValueError(f"Expected one Python symbol {symbol} in {path}; found {len(candidates)}")
        start, end = candidates[0].lineno, candidates[0].end_lineno
    elif symbol:
        owner, separator, name = symbol.rpartition("::")
        pattern = re.compile(r"\b(?:fn|struct|enum)\s+" + re.escape(name if separator else symbol) + r"\b")
        scope = None
        candidates = []
        for i, line in enumerate(lines):
            implementation = re.match(r"^impl\s+(\w+)\s*\{", line)
            if implementation:
                scope = implementation.group(1)
            if pattern.search(line) and (not separator or scope == owner):
                candidates.append(i + 1)
        if len(candidates) != 1:
            raise ValueError(f"Expected one Rust symbol {symbol} in {path}; found {len(candidates)}")
        start = end = candidates[0]
    elif "contains" in reference:
        candidates = [i + 1 for i, line in enumerate(lines) if reference["contains"] in line]
        if not candidates:
            raise ValueError(f"Source anchor missing in {path}: {reference['contains']}")
        start = end = candidates[0]
    else:
        start, end = 1, len(lines)
    return {**reference, "line": start, "end_line": end, "sha256": hashlib.sha256(location.read_bytes()).hexdigest(),
            "url": "source.html?path=" + path.as_posix() + "#L" + str(start)}


def discovery(root):
    """Return lexical imports, not inferred semantic relationships."""
    result = []
    directories = ("chromatensor", "experiments", "dashboard", "scripts", "native", "tests", "requirements", ".github")
    paths = [root / "pyproject.toml"]
    paths.extend(path for path in (root / "native/rust-notices").rglob("*") if path.is_file())
    for directory in directories:
        paths.extend(path for path in (root / directory).rglob("*") if path.is_file()
                     and path.suffix in {".py", ".rs", ".h", ".js", ".html", ".css", ".toml", ".lock", ".yml", ".yaml", ".txt"}
                     and not {"target", "__pycache__", "node_modules", ".venv"}.intersection(path.parts))
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        imports = []
        if path.suffix == ".py":
            for item in ast.walk(ast.parse(text)):
                if isinstance(item, ast.Import):
                    imports.extend({"module": alias.name, "line": item.lineno} for alias in item.names)
                elif isinstance(item, ast.ImportFrom):
                    imports.append({"module": "." * item.level + (item.module or ""), "line": item.lineno})
        elif path.suffix == ".rs":
            imports = [{"module": line.strip(), "line": i + 1} for i, line in enumerate(text.splitlines()) if line.strip().startswith("use ")]
        result.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "imports": imports})
    return result


def validate_graph(graph, root, verify_sources=True):
    ids = [item["id"] for item in graph["nodes"]]
    groups = {item["id"] for item in graph["groups"]}
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate component ID")
    edge_ids = [item["id"] for item in graph["edges"]]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate relationship ID")
    for item in graph["nodes"]:
        if REQUIRED_NODE_FIELDS - item.keys() or item["group"] not in groups:
            raise ValueError(f"Incomplete component: {item['id']}")
        expected_status = "experimental" if item["group"] in {"legacy", "research"} else "implemented"
        if item["status"] != expected_status:
            raise ValueError(f"Component status differs from reviewed classification: {item['id']}")
        if not item["sources"] or not item["tests"]:
            raise ValueError(f"Component requires source and verification references: {item['id']}")
        if set(item["dependencies"]) - set(ids):
            raise ValueError(f"Unknown dependency: {item['id']}")
    for item in graph["edges"]:
        if item["source"] not in ids or item["target"] not in ids or item["type"] not in EDGE_TYPES or not item["evidence"]:
            raise ValueError(f"Invalid semantic relationship: {item['id']}")
        if item["source"] == item["target"]:
            raise ValueError(f"Self relationship requires review: {item['id']}")
    reviewed = {item["id"]: item for item in EDGES}
    if set(edge_ids) != set(reviewed):
        raise ValueError("Semantic relationship set differs from the reviewed registry")
    for item in graph["edges"]:
        expected = reviewed[item["id"]]
        if any(item[key] != expected[key] for key in ("source", "target", "type", "label", "boundary", "flows")):
            raise ValueError(f"Semantic relationship differs from review: {item['id']}")
        anchors = [{key: reference[key] for key in ("path", "symbol", "contains") if key in reference} for reference in item["evidence"]]
        if anchors != expected["evidence"]:
            raise ValueError(f"Semantic relationship evidence differs from review: {item['id']}")
    for flow in graph["flows"]:
        if set(flow["steps"]) - set(ids):
            raise ValueError(f"Unknown journey component: {flow['id']}")
    flow_ids = {flow["id"] for flow in graph["flows"]}
    for item in graph["edges"]:
        if set(item["flows"]) - flow_ids:
            raise ValueError(f"Unknown relationship journey: {item['id']}")
    if verify_sources:
        for reference in all_references(graph):
            resolved = resolve_reference(root, {key: reference[key] for key in ("path", "symbol", "contains") if key in reference})
            if any(reference[key] != resolved[key] for key in ("line", "end_line", "sha256", "url")):
                raise ValueError(f"Source reference drift: {reference['path']}")
    # Semantic claims are curated, not import-derived. Guard the most consequential boundaries.
    pairs = {(edge["source"], edge["target"]): edge for edge in graph["edges"]}
    required = {("python", "ffi"), ("ffi", "storage"), ("browser", "http"), ("http", "inspection"), ("datasets", "training")}
    if not required.issubset(pairs) or any(not pairs[pair]["boundary"] for pair in required):
        raise ValueError("Required native, HTTP or dataset trust boundary missing")
    if ("browser", "operators") in pairs or ("nn", "operators") in pairs:
        raise ValueError("Legacy execution must not claim native operator dispatch")
    return True


def all_references(graph):
    for item in graph["nodes"]:
        yield from item["sources"]
        yield from item["tests"]
    for item in graph["edges"]:
        yield from item["evidence"]
    for item in graph["integrations"]:
        yield from item["sources"]


def build_graph(root=ROOT):
    graph = json.loads(json.dumps(dict(schema_version=1, groups=GROUPS, nodes=NODES, edges=EDGES, flows=FLOWS, integrations=INTEGRATIONS)))
    for item in graph["nodes"]:
        item["status"] = "experimental" if item["group"] in {"legacy", "research"} else "implemented"
        item["sources"] = [resolve_reference(root, source) for source in item["sources"]]
        item["tests"] = [resolve_reference(root, source) for source in item["tests"]]
        item["dependencies"] = sorted({edge["target"] for edge in graph["edges"] if edge["source"] == item["id"] and edge["type"] == "call"})
    for item in graph["edges"]:
        item["evidence"] = [resolve_reference(root, source) for source in item["evidence"]]
    for item in graph["integrations"]:
        item["sources"] = [resolve_reference(root, source) for source in item["sources"]]
    graph["inventory"] = discovery(root)
    graph["source_identity"] = hashlib.sha256(json.dumps(graph["inventory"], sort_keys=True).encode()).hexdigest()
    graph["evidence_note"] = "Source links resolve to generated snapshots of actual local files. Hash and symbol checks detect drift. Tests listed here identify relevant verification code, not a claim that those tests passed in this run. Semantic edges are reviewed assertions, separate from lexical import discovery. Absent integrations describe this reviewed source, not future plans."
    graph["semantic_review"] = {"method": "Manually curated source behavior and ownership contracts, with exact source anchors. Validation protects the reviewed registry from stale or altered graph data; it does not automatically prove program behavior.", "registry": "scripts/architecture.py", "automatic_discovery_creates_edges": False}
    graph["status_note"] = "Implemented means the reviewed source has a runnable implementation, not that current release gates passed. Experimental marks numerical research APIs and experiments. Absent means no implementation or integration was found in the reviewed runtime. This map has no event bus or streaming edge because no such runtime path was found."
    validate_graph(graph, root)
    return graph


def generated_files(root=ROOT):
    graph = build_graph(root)
    snapshots = {reference["path"]: {"sha256": reference["sha256"], "text": (root / reference["path"]).read_text(encoding="utf-8-sig")}
                 for reference in all_references(graph)}
    serialized = json.dumps(graph, indent=2, ensure_ascii=True) + "\n"
    return {"graph.json": serialized, "graph.js": "window.CHROMATIC_ARCHITECTURE = " + serialized.rstrip() + ";\n",
            "sources.js": "window.CHROMATIC_SOURCES = " + json.dumps(snapshots, ensure_ascii=True) + ";\n"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Regenerate static graph and source snapshots, the default action")
    parser.add_argument("--check", action="store_true", help="Reject stale generated data, source drift and invalid semantic graph")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.build and args.check:
        parser.error("Use --build or --check, not both")
    root = args.root.resolve()
    output = root / OUTPUT
    expected = generated_files(root)
    if args.check:
        for name, text in expected.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Stale architecture output: {path}. Run python scripts/architecture.py")
        stored = json.loads((output / "graph.json").read_text(encoding="utf-8"))
        validate_graph(stored, root)
        print(f"Architecture checked: {len(stored['nodes'])} components, {len(stored['edges'])} reviewed relationships, {len(stored['inventory'])} inventoried files")
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, text in expected.items():
            (output / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"Architecture generated in {output}. Open index.html or serve the repository root.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, SyntaxError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
