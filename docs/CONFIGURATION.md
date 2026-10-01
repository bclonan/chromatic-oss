# Configuration

No API key, cloud credential, production account or `.env` file is required. The application reads no project-specific secret environment variables. Do not create fake provider settings for services the code does not use.

| Setting | Default | Effect and safe example |
| --- | --- | --- |
| Dashboard `--host` | `127.0.0.1` | Local bind address. Keep loopback; no authentication or TLS proxy is supplied. |
| Dashboard `--port` | `8765` | HTTP port; use `--port 8767` for a conflict. |
| Dashboard `--results` | `results/latest.json` below the working directory | JSON result to display. Use `--results results/local-smoke.json`. |
| Benchmarks `--profile` | `pilot` | `smoke`, `pilot` or `full`; changes actual task counts and training budget. |
| Benchmarks `--device` | CUDA if available, otherwise CPU | Use `--device cpu` for the dependency locks. CUDA needs an independently installed compatible Torch build and device. |
| Benchmarks `--seeds` | `7 23` | Explicit seeds; use `--seeds 7` for a small local run. |
| Benchmarks `--tasks` | All defined task names | Scalar, actual MNIST/CIFAR, Tiny Shakespeare language and generated periodic, precision, operators and attention. Select a subset explicitly. |
| Benchmarks `--representations` | Defined research representations | Compare equal bit budgets; `--help` lists the current choices. |
| Benchmarks `--epochs`, `--precision-samples` | Profile-dependent | Override measured work; the config and receipt retain the values. |
| Benchmarks `--output` | `results/latest.json` | A local JSON output. Choose a distinct filename for altered configurations. |
| Benchmarks `--resume` | Off | Resume only a matching complete configuration; incompatible work fails validation. |
| Native builder `--platform` | `host` | `linux` uses a pinned Rust Docker image on Windows. |
| Native builder `--wheel-dir` | `artifacts/packages/native-hue` | Output directory inside this repository so provenance can verify its artifacts. |
| Native builder `--test` | Off | Run Rust unit and CLI tests before building. |
| Native verifier `--samples`, `--seed`, `--repeats` | `1000000`, `7331`, `3` | Independent integer oracle and measurements; reduced values do not prove the million-token gate. |
| Native verifier `--output` | `results/native-hue-verification.json` | Candidate proof receipt. |
| Cube experiment `--seeds`, `--epochs` | `7 23`, `24` | Seeded generator and training count; reduced runs are labeled as such. |
| Cube experiment `--output` | `results/cube.json` | Result displayed by Cube Lab. |
| Train `--representation`, `--task`, `--seed`, `--profile` | `chroma`, `scalar`, `7`, `pilot` | Single research training job, distinct from native hue arithmetic. |
| `CUBLAS_WORKSPACE_CONFIG` | Set to `:4096:8` if absent | Deterministic CUDA workspace convention in experiments/tests. Not needed for native CPU operators. |
| `CARGO_TARGET_DIR` | Build-script artifact directory | Cargo output location. The builder sets this for isolated outputs. |
| `PYTEST_ADDOPTS` | Pytest's normal environment behavior | The evidence runner removes this variable to ensure complete collection. |

All scripts expose their remaining verification, output and internal cold-process arguments through `--help`. Native file APIs expose `max_bytes` and enforce a fixed `MAX_ELEMENTS` allocation limit. The legacy loaders expose their documented byte budgets. Consult [the native guide](NATIVE_HUE.md) and [file formats](FILE_FORMATS.md) for their current values and semantics. Ordinary package managers may read their own proxy, index and cache environment settings; those are not application credentials.

Storage is local. Dataset loaders create `data/`. Experiments create checkpoint and measurement files under `artifacts/` and `results/`. Tests and independent package matrices create ignored environments and scratch directories. No scheduled retention job deletes them.
