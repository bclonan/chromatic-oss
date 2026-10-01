# Security policy

ChromaTensor is a local research library and inspector. It has no accounts, authentication service, billing, customer records, or hosted production backend. The dashboard defaults to `127.0.0.1`. Keep it on a trusted local machine.

The dashboard checks the request Host and the Origin of computation requests, bounds JSON bodies and tensor sizes, and adds browser security headers. Those checks do not authenticate local processes. Binding it to a network address exposes recorded results and artifact downloads to anyone who can reach that address. The dashboard has no deployment authentication, rate limiter, or global concurrency budget. Do not expose it as a public service without separate deployment controls and review.

Public tensor loaders validate metadata, payload lengths, archive members, numeric dtypes, and checksums. They avoid executable pickle loading. Use the documented byte limits in [file formats](docs/FILE_FORMATS.md) when loading unfamiliar files. A checksum detects changes, it does not establish who created a file. File and expansion limits do not cap parser CPU time, peak process memory, or GPU allocation. Load hostile files in an isolated process with operating system resource limits.

Research commands download MNIST, CIFAR-10, or Tiny Shakespeare when those tasks request them. They write local datasets, models, reports, and figures. Inspect results before sharing them. Provenance can record dependency versions, Git state, local paths, and hardware. Do not include customer data, credentials, browser profiles, or private logs in experiment inputs or release evidence.

## Reporting a vulnerability

Use the repository host's private vulnerability reporting channel when the maintainer enables it. On GitHub, use Security, then Report a vulnerability. Include the affected version, source location, a minimal synthetic example, and expected impact. Do not post live secrets or private data in a public issue. If no private channel is available, ask the maintainer for one without disclosing exploit details publicly.

No response time or security support period is promised. The maintainer must configure a private reporting channel before publication.

## Release checks

Run the privacy scanner against the reviewed manifest, scan with Gitleaks, and inspect generated packages and archive members before a release. [Security review](docs/SECURITY_REVIEW.md) records the current scope and limits. Reports contain finding kinds and source locations only. Never print credential values. If a credential is exposed, revoke or rotate it through its provider. Removing a file or starting fresh Git history does not revoke a credential.
