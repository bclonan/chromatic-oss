# Development

The primary tensor is Rust code exposed through a C ABI and Python `ctypes`. The separate `chromatic-hue` wheel has no third-party runtime dependencies. The `chromatensor` research package retains the PyTorch codecs, neural layers, cubes, experiments and local dashboard.

Use Python 3.11 through 3.14 and Rust 1.96.0 on x86-64 Windows or Linux. Windows Rust builds require the MSVC build tools and Windows SDK. Linux builds require a linker and libc development files. Docker is optional for local use and required by the Windows-driven Linux package matrix. macOS and ARM builds are not verified release targets.

## Native-only development

From the repository root:

```powershell
python -m venv .venv-hue
python scripts/build_native_hue.py --test
.venv-hue/Scripts/python.exe -m pip install --no-deps artifacts/packages/native-hue/chromatic_hue-0.1.0-py3-none-win_amd64.whl
.venv-hue/Scripts/python.exe -I scripts/verify_installed_hue.py --output artifacts/local-native-install.json
```

Use `.venv-hue/bin/python` and the `linux_x86_64` wheel on Linux. Linux wheels retain their measured glibc boundary and are not claimed to be manylinux wheels. The build script writes the source binding's binary under `chromatensor/_native/`, the wheel under `artifacts/packages/native-hue/`, and a receipt under `results/`.

## Research development on Windows

Install `uv`, then create an independent CPU environment. The lock pins exact dependency versions for Python 3.12.

```powershell
uv venv --python 3.12 --seed .venv
uv pip sync --python .venv/Scripts/python.exe requirements/python312-cpu.lock --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --strict
uv pip install --python .venv/Scripts/python.exe --no-deps --no-build-isolation -e ".[test]"
uv pip check --python .venv/Scripts/python.exe
.venv/Scripts/python.exe scripts/build_native_hue.py --test
.venv/Scripts/python.exe -m pytest tests -q
```

The two package indexes are needed because the recorded Torch wheels have a `+cpu` suffix. The version locks do not contain wheel download hashes. On Linux choose the matching `requirements/linux312-cpu.lock` and `.venv/bin/python`. Other Python versions have their own locks. Python 3.11 uses NumPy 2.4.6 because NumPy 2.5 requires Python 3.12. Torchvision excludes Python 3.14.1; use a current patched Python 3.14 interpreter for research. Cross-platform instructions and execution evidence are listed separately in the [release audit](RELEASE_AUDIT.md).

The research wheel does not contain a native library. Install the platform-specific `chromatic-hue` wheel alongside it to use its native `Tensor` API outside a source build. The legacy scalar APIs remain explicit imports. Do not relabel them as native hue arithmetic.

## Code map and extension points

See [the repository map](REPOSITORY_MAP.md) and [architecture](../ARCHITECTURE.md). Extend native operations in `native/hue_tensor/src/lib.rs`, then update `ffi.rs`, the header, `chromatensor/hue.py`, Rust tests and Python tests together. All constructors validate shape and allocation budgets before copying user input. Keep handle lifetime protection around every native call.

Add numerical codecs through `chromatensor/codecs/base.py` and preserve their serialized configurations. Extend file formats only with version checks and bounded loading. Add experiments through the existing dataset, model, measurement and provenance code. Compare equal bit budgets and record actual execution precision. Results use adjacent temporary files and replacement retries so a reader never sees partial JSON.

The dashboard has no database, migration, model provider, login, payment processor, job queue or cloud service. It reads local JSON and allowed artifacts. Its POST handlers run bounded inspection work synchronously. Do not expose this development server publicly.

## Debugging

If the native library is missing, build it before importing a native tensor or install the separate native wheel. If the MSVC linker is missing, install the Rust platform toolchain prerequisites. If a saved receipt fails current-source validation, regenerate it; do not edit its source hash to force a pass. If dataset downloads fail, repair connectivity or provide the actual verified dataset cache. If port 8765 is occupied, pass `--port 8767` to `scripts/dashboard.py`.

Test scratch directories live under `.pytest-work/` with unique names. CUDA-specific cases can skip in CPU environments. The [testing guide](TESTING.md) records what was actually executed.
