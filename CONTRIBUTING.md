# Contributing

Read [the developer guide](docs/DEVELOPER_GUIDE.md), [AGENTS.md](AGENTS.md) and the [release decisions](docs/DECISIONS.md). Keep changes focused and explain their observable behavior. Add regression tests for numerical, trust-boundary and lifecycle changes. Run the relevant tests, then the full suite before proposing a release.

Preserve actual color bytes, mathematical contracts, seeds, dataset provenance, sample counts, storage costs and execution precision. Use equal bit budgets for comparisons. Do not describe floating-point quantization-aware training as integer or entirely chromatic execution. Record failed experiments and repairs in `RESEARCH_LOG.md`; update `TODO.md` and `EXPERIMENT_MATRIX.md` from actual receipts.

Do not commit credentials, personal paths, raw downloaded datasets, local environments, browser profiles, user uploads or machine logs. Run the release scanner and architecture drift check before submitting source changes. See [security guidance](SECURITY.md) for vulnerability reporting.

Contributions to the public version must be compatible with AGPL-3.0. Maintainer-owned code can also be offered under a separate paid agreement. A third-party contribution does not automatically grant the maintainer authority to sell exceptions to that contributor's copyright. Resolve that permission before accepting a contribution intended for commercial relicensing. No contributor agreement is invented here.
