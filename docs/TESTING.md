# Testing and troubleshooting

Candidate execution records live in [the release audit](RELEASE_AUDIT.md) and `docs/release/verification.json`. Historical receipts do not certify changed source. There is no external model/provider integration.

Run after [setup](DEVELOPER_GUIDE.md):

```powershell
.venv/Scripts/python.exe scripts/build_native_hue.py --test
.venv/Scripts/python.exe -m pytest tests -q --junitxml=artifacts/release-tests.xml
cargo fmt --manifest-path native/hue_tensor/Cargo.toml --check
cargo clippy --offline --manifest-path native/hue_tensor/Cargo.toml --all-targets -- -D warnings
python scripts/architecture.py --build
python scripts/architecture.py --check
python scripts/release_audit.py --check-docs
python scripts/scan_release.py
```

Pytest covers codecs, operators, storage, native handle lifetime, CLI behavior, dashboard HTTP protections, checkpoint/provenance validation and release tooling. CUDA-specific cases can skip in CPU environments. Record them separately. Rust tests exercise arithmetic and malformed files; Clippy denies warnings. Architecture validation checks paths, symbols, hashes, assets and graph relationships. It cannot infer arbitrary runtime behavior.

## Native proof and packages

```powershell
.venv/Scripts/python.exe scripts/verify_native_hue.py --samples 1000000 --seed 7331 --repeats 3
python scripts/build_native_hue.py --platform linux --test
python scripts/run_native_hue_package_checks.py
```

The proof uses NumPy as an independent equal-32-bit oracle. Native operators and the cold process do not use NumPy or PyTorch. The matrix driver runs on Windows, uses `uv` for its Windows interpreters, and requires Docker in Linux container mode. It tests new wheels in clean Python 3.11 through 3.14 environments. Prepare missing Linux images before running it:

```powershell
foreach ($version in @('3.11', '3.12', '3.13', '3.14')) {
    $tag = "chromatensor-review-ci:python-$version"
    docker image inspect $tag *> $null
    if ($LASTEXITCODE -ne 0) {
        "FROM python:$version-slim-bookworm" | docker build --platform linux/amd64 --tag $tag -
        if ($LASTEXITCODE -ne 0) { throw "Failed to prepare $tag" }
    }
}
```

These official Python base images need network access when first pulled. The verification containers run with networking disabled and create a separate dependency-free environment. Existing images remain unchanged. The candidate run used already prepared images, with interpreter versions and image IDs in its receipt. The preparation recipe above is reproduction guidance, not a claim that new images were built during this release. These native probes do not certify the full research dependency matrix. Native source builds also work directly on the tested Linux target.

`python scripts/verify.py` checks six source-bound native gates. Regenerate all inputs against frozen source before running it. Do not edit hashes or call missing evidence a pass. `--legacy` requires full trained research and browser receipts, not a smoke run. The release audit records those scopes independently.

## Research and browser checks

```powershell
.venv/Scripts/python.exe scripts/run_benchmarks.py --profile smoke --device cpu --seeds 7 --tasks scalar periodic precision operators attention --output results/local-smoke.json
.venv/Scripts/python.exe scripts/run_cube_experiments.py --seeds 7 --epochs 2 --output results/cube.json
.venv/Scripts/python.exe scripts/dashboard.py --results results/local-smoke.json
```

Open `http://127.0.0.1:8765`. Check recorded results, tensor inspection, invalid-input feedback, cube turns/undo and spectral erasure. Open the explorer through README's server command. Check flow selection, source search, filters, groups, zoom, dragging, fit/reset, keyboard operation and details/source links. Browser checks establish those interactions only, not universal assistive-technology usability.

Repair missing libraries or artifacts rather than skipping required tests. Preserve old receipts and regenerate candidate evidence after changes. Use matching platform/Python locks and the official CPU index for Torch. Keep CPU and CUDA evidence separate. Review memory costs before raising file or HTTP budgets.
