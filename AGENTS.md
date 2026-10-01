# ChromaTensor working rules

Apply the user's Unslop rules to prose. Use plain language and straight quotes. Avoid em dashes and decorative formatting.

Run tests after changing numerical behavior. Preserve seeds, dataset provenance, sample counts, measured storage, and execution precision in reports. Compare equal bit budgets. Do not label floating point quantization-aware training as integer or entirely chromatic execution. Never replace real datasets with synthetic data under the same experiment name.

Keep required paths runnable. Do not add unfinished implementations. Record failures and repairs in RESEARCH_LOG.md. Update TODO.md and EXPERIMENT_MATRIX.md against execution evidence.

The primary native API stores canonical RGBD bytes and computes in Z/391680Z. Keep that arithmetic separate from the legacy scalar RGBD format and floating-point PyTorch research. Native D is a sub-hue position; legacy D is an IEEE exponent. Preserve bounded parsing, allocation checks, native handle ownership and actual storage costs.

Use docs/DEVELOPER_GUIDE.md for current setup, docs/TESTING.md for checks and docs/RELEASE_GUIDE.md for packaging. Older research documents and sanitized receipts are historical evidence. Do not claim they validate changed source. Run the full tests after numerical changes, the release privacy scan before selecting public contents and scripts/architecture.py --build then --check after source changes.

Release files come from docs/release/manifest.json and scripts/release_audit.py. Never use ignore rules alone as the privacy policy. Keep raw machine logs, credentials, downloaded datasets, probe arrays, local caches and original Git history out of release selection. Do not push, publish or send communications without explicit authorization.

Copyright belongs to Brad Clonan where established by this project. The approved public license is AGPL-3.0-only. Preserve LICENSE, COPYRIGHT and THIRD_PARTY_NOTICES.md in distributed packages. A paid commercial exception requires a separate maintainer agreement. Do not invent an exception or assign contributor copyrights.

