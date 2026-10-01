# Release decisions

These decisions were made during public-release preparation on 2026-10-01. They are not historical design rationale.

| Decision | Reason | Consequence |
| --- | --- | --- |
| Preserve native and legacy implementations | Both are implemented and represent different arithmetic contracts | Document separate APIs and evidence; no feature removal to obtain passing checks |
| Create a sibling release repository | The maintainer required original history and work to remain intact | Copy an explicit reviewed manifest and initialize new Git history |
| Use AGPL-3.0 and offer paid exceptions | Maintainer selected this model after reviewing its commercial-use boundary | Compliant commercial use is allowed without payment; separate commercial terms need a maintainer agreement |
| Generate a static offline architecture explorer | Architecture should work without credentials or backend services | Bundle assets and source snapshots; validate drift against selected source |
| Separate historical and candidate receipts | Public sanitization and repairs change file identities | Historical metrics remain evidence of their original runs; rerun candidate checks |
| Omit private caches and dataset snapshots | They are unnecessary for source installation and can contain private paths or redistribution obligations | Retain code, seeds and regeneration procedures; do not replace actual datasets with synthetic data |

Protect the finite hue ring semantics, actual four-byte storage, bounded loading, handle ownership, equal bit budgets and scientific provenance. A successful codec round trip does not establish a learning or speed advantage. Native execution is CPU execution. Legacy neural training retains floating parameters, gradients and dense kernels.
