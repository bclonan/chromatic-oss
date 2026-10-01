# Repository map

| Path | Responsibility |
| --- | --- |
| `native/hue_tensor/src/lib.rs` | Owned RGBD tensors, finite-ring arithmetic, shape operations and validation |
| `native/hue_tensor/src/ffi.rs` | C ABI, handle lifetime, errors and pointer boundaries |
| `native/hue_tensor/src/storage.rs` | Bounded native file format, checksum and temporary writes |
| `native/hue_tensor/src/bin/hue_tensor.rs` | Dependency-free CLI |
| `native/hue_tensor/include/hue_tensor.h` | Public C declarations |
| `chromatensor/hue.py` | Standard-library Python native binding and encoder/decoder API |
| `chromatensor/__init__.py` | Primary native aliases and lazy legacy imports |
| `chromatensor/tensor.py`, `codecs/`, `operators/` | Explicit earlier numerical tensor implementation |
| `chromatensor/serialization/` | Versioned checked legacy persistence |
| `chromatensor/cube.py`, `holographic.py` | Integer geometry and floating-point software Fourier reconstruction |
| `chromatensor/autograd/`, `nn/` | Floating-point quantization-aware neural research |
| `experiments/` | Datasets, seeded training, controls, measurement and provenance |
| `dashboard/` | Local browser UI and bounded inspection server |
| `scripts/` | Build, test, research, native proof and package verification commands |
| `tests/` | Meaningful numerical, format, security, dashboard and release regressions |
| `requirements/` | Platform/Python dependency locks and minimum versions |
| `docs/architecture/` | Static interactive explorer and structured graph |
| `results/`, `artifacts/` | Selected historical measurements and generated candidate outputs |

Accounts, guest passes, billing, uploads, chats, project ownership, Convex, inference providers, webhooks and databases are absent. The legacy dashboard accepts numeric inspection inputs; it does not upload files to a provider. See [architecture](../ARCHITECTURE.md) for actual boundaries and source references.
