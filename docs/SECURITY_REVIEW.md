# Security and public-content review

Review date is October 1, 2026. This review concerns a local research source release. It does not certify the library as secure, production-ready, or safe for public hosting.

## Scope and findings

The original checkout had 821 tracked files, no untracked files, and 155,912 ignored files. Tracked contents totaled 689,614,008 bytes. The review inventoried ignored environments, test directories, build products, downloaded datasets, caches, and generated outputs. Their contents were excluded from the public selection. No selected ignored configuration, browser profile, uploaded data, or credential store was found.

The expanded privacy scan inspected all 821 tracked files as bytes. It inspected ZIP members without executing or deserializing them, including wheels, numeric archives, and checkpoint containers. There were no scan budget limits. It found 983 personal filesystem paths in generated evidence and logs. It found no private-key, hosted-token, credential-URL, or credential-assignment patterns in the source review. The initial selected project-source review found one email address in a synthetic test Git identity, `evidence@example.invalid`. It remains intact. The release later added official Rust notices. Their legitimate upstream authors, copyright statements, and intentionally public contact addresses remain unchanged.

The original history contained seven commits. Gitleaks's default scan returned 292 generic API-key findings. Reviewers retrieved every matching historical line and confirmed that each value was a 64-hex SHA-256 provenance digest under one of four filename keys. Those are false positives, with no credential rotation indicated. The committed Gitleaks configuration permits only the exact reviewed filename and SHA-256 match shape. It does not exempt directories or arbitrary hexadecimal strings. The second seven-commit scan passed.

There were no tracked notebooks, Git submodule links, `.gitmodules`, or `.gitattributes`. No Git LFS configuration or pointer files were selected. The 40 tracked image files had no EXIF entries. Manual pixel review of the four original dashboard screenshots and native color-tensor diagram found no credentials, personal browser chrome, contact details, or private paths. Plot metadata and generated diagrams identify project output. The original cube screenshot is mostly blank, so it is not strong evidence of rendering correctness.

The release added two dashboard screenshots, `artifacts/dashboard/architecture-explorer.jpg` and `artifacts/dashboard/release-cube.jpg`. Both have zero EXIF entries and passed a separate pixel review for private information. The final replacement cube image shows colored cubies, labeled axes, and projected slices.

Historical datasets, dataset-derived input arrays, checkpoint binaries, executable files, native libraries, old wheels, raw logs, build metadata, caches, and original Git history are excluded. Reviewed historical JSON and Markdown reports can preserve measured results after path sanitization. They record the original file hash and the sanitization treatment. They are historical observations. Their old source and artifact verification does not transfer to edited release source, and excluded artifacts cannot be revalidated from those reports alone.

Private raw reports stay outside the public repository. Public summaries include counts, finding categories, safe source locations, commands, and limitations. They contain no credential values or private email addresses.

An intermediate precommit scan found 29 private-user-path pattern matches in `results/local-smoke.json`. A fresh smoke run replaced that output, and the next selected-content scan passed with 335 files, zero findings, and no scan limits. The earlier output bytes were unavailable when reviewers tried to classify those fields. The failure remains recorded without a claim about what caused the old matches.

A clean clone initially changed Rust notice line endings through the host's Git configuration. Native build verification rejected the changed notice hashes. The release adds `.gitattributes` with `* -text` so clones preserve the selected file bytes. This matters for both attribution and verification.

## Tools and execution

| Check | Tool and version | Executed scope | Result |
| --- | --- | --- | --- |
| Original inventory | Git and Python 3.14 | Tracked, untracked, ignored names, sizes, and modes | PASS, original preserved |
| Original expanded privacy scan | `scripts/scan_release.py` 1.0 | 821 tracked files, 689,614,008 bytes, ZIP depth 3 | FAIL, 983 paths to sanitize or exclude |
| Initial source secret scan | Gitleaks 8.30.1 default rules | Initial selected source, 1,329,900 bytes | PASS, zero findings |
| Original history default secret scan | Gitleaks 8.30.1 | All seven commits, 204,761,504 scanned bytes, archive and decode depth 3 | FAIL, 292 findings subsequently classified as provenance hashes |
| Original history reviewed secret scan | Gitleaks 8.30.1 and exact match allowlist | Same seven commits | PASS, zero remaining findings |
| Scanner negative controls | pytest in release environment | Eleven privacy, manifest, redaction, archive, and upstream attribution cases | PASS |
| Active dependency advisory scan before repair | pip-audit 2.10.1, PyPI service | Release environment | FAIL, two unique setuptools advisories |
| Active dependency advisory scan after repair | pip-audit 2.10.1, OSV service | 30 queried distributions, project editable package skipped | PASS, zero known advisories |
| Independent clean-clone installed dependency scan | pip-audit 2.10.1, OSV service | 32 distributions in Python 3.12, exact installed CPU versions | PASS, zero known advisories and zero skipped packages |
| Active dependency license metadata | Python distribution metadata | Python 3.14 CPU lock, 29 packages | PASS, complete metadata inventory |
| Current dependency specifications | pip-audit 2.10.1, OSV service | All ten committed requirements files, CPU version suffix normalized | PASS, zero findings and zero skipped packages |
| Candidate wheel privacy and notices | Release byte scan 1.0 and exact byte comparisons | Research wheel, Windows native wheel, Linux native wheel | PASS, zero sensitive patterns or archive limits, bundled notices match source |
| Independent clean-clone wheel privacy and notices | Release byte scan 1.0 and exact byte comparisons | Independently built research wheel and Windows native wheel | PASS, zero sensitive patterns or archive limits, declared licenses and bundled notices match cloned source |
| Selected public-content staging | Release byte scan 1.0 and Gitleaks 8.30.1 | 349 reviewed manifest entries plus the manifest itself, 350 staged files | PASS, zero privacy or secret findings and no privacy scan limits |
| Final amended history and release archive | Release scanners and exact manifest comparisons | Run after the final audit-document amend and ZIP construction | The final handoff records these immutable checks; receipts remain outside the archive |

Gitleaks 8.30.1 came from its [official release](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1). The Windows archive's SHA-256 matched the published checksum, `d29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e`. The scan uses default rules plus the disclosed provenance exception. [Gitleaks documentation](https://github.com/gitleaks/gitleaks) describes its Git, directory, redaction, and archive options.

The advisory scanner installed in a separate private tooling environment. It did not add dependencies to the project environment. PyPI reported CPU builds of PyTorch and torchvision as unavailable for its version lookup. An independent OSV query covered those exact installed CPU versions. That coverage difference remains explicit in the evidence.

The [clean-clone dependency audit](evidence/clean-clone-dependency-audit.json) observes its separate Python 3.12 environment after the clone checks. OSV queried all 32 installed distributions, including exact CPU versions of PyTorch and torchvision and setuptools 83.0.0. It returned no known advisories and skipped no packages.

The old supported minimum pins also produced known advisories in PyTorch, pytest, and setuptools. The release raises its dependency floors and replaces those minimum pins. The [dependency audit](evidence/dependency-audit.json) records the SHA-256 of each of the ten current requirement files, every audit exit code, normalization, and coverage limits. No unlisted transitive resolution or installation is implied by auditing a requirement file. The active environment scan separately observes installed versions.

The [release wheel inspection](evidence/package-security.json) covers all three selected wheels. It verifies AGPL-3.0-only metadata, every declared project and Rust notice file, and source and native-library correspondence within each build. The [clean-clone inspection](evidence/clean-clone-package-security.json) repeats those checks for the two wheels built independently from the clone. Each check compares a wheel with its own build's files. It does not establish bit-identical output between builds.

Setuptools 83.0.0 addresses the observed [PackageIndex path traversal](https://github.com/pypa/setuptools/security/advisories/GHSA-5rjg-fvgr-3xxf) and [source archive exclusion issue](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c). Those fixes matter to build tools even though the dashboard does not call those APIs. Rebuilding with the upgraded tool is part of release verification.

## Repeatable commands

Run these commands from the release root. Use the final reviewed manifest when selecting public contents. A Git-mode privacy scan checks tracked files, so stage or commit the intended files first when no manifest argument is used.

```powershell
python scripts/scan_release.py --manifest docs/release/manifest.json --output docs/evidence/privacy-scan.json
python scripts/scan_release.py --licenses --lock requirements/python314-cpu.lock --output docs/evidence/dependency-licenses.json
python -m pytest tests/test_release_scan.py -q
gitleaks dir . --redact=100 --no-banner --report-format=json --report-path=../private-gitleaks-files.json --max-archive-depth=3 --max-decode-depth=3
gitleaks git . --log-opts="--all" --redact=100 --no-banner --report-format=json --report-path=../private-gitleaks-history.json --max-archive-depth=3 --max-decode-depth=3
```

Run directory scans on a manifest-built staging directory, or inspect exclusions before scanning a working tree with generated outputs. Do not publish a scanner's raw report just because secret strings were redacted. Git reports can still contain private author metadata.

To inspect a produced ZIP or wheel, add `--archive` with that file's path to the privacy command. The scanner reads nested ZIP members without extracting files. Gitleaks covers other supported archive encodings. For pip-audit, install the tool in a separate environment and use `--path` to the project's `site-packages`, `--skip-editable`, `--vulnerability-service osv`, and `--format json`. [pip-audit documentation](https://github.com/pypa/pip-audit) describes that mode.

## Limits and release decisions

Secret detection and pattern checks do not prove absence of private information. Binary inspection is not malware analysis. Image metadata and selected pixel review do not detect hidden steganographic content. The review did not validate credentials against providers, scan every excluded local environment byte, or test a hosted service. No exposed credential was confirmed and no rotation was performed.

The Python license inventory observes the active environment's installed packages. It does not verify every platform wheel in every requirement lock, determine legal compatibility, or establish rights to datasets or trained model output. No third-party dependency or dataset binary is vendored in the source release. See [third-party notices](../THIRD_PARTY_NOTICES.md).

The user authorized AGPL-3.0-only and allowed a public GitHub noreply identity. Existing author and remote evidence establish Brad Clonan and the public handle `bclonan`. The new repository may use `Brad Clonan <bclonan@users.noreply.github.com>` locally. The review does not expose the original private email addresses, change global Git configuration, copy old history, or authorize publication.

The local dashboard has no authentication or service-wide resource limit. Public hosting needs a separate deployment design and review. The maintainer must enable a private vulnerability reporting channel on the chosen repository host. Paid licensing exceptions require separate agreements with the relevant copyright holders.
