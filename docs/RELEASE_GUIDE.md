# Build and release

The reviewed documentation and source ZIP are published through the [static documentation site](HOSTING.md). Original repository history is excluded. The supported local product is a native CPU library and CLI, plus the legacy numerical research package and loopback dashboard. It needs no hosted infrastructure, database migration, model provider or payment sandbox.

## Build packages

Follow [development setup](DEVELOPER_GUIDE.md), then run:

```powershell
python scripts/build_native_hue.py --test
.venv/Scripts/python.exe -m pip wheel . --no-deps --no-build-isolation --wheel-dir artifacts/packages
```

Native package `chromatic-hue` is version 0.1.0. Research package `chromatensor` is version 0.2.0. The research wheel contains Python modules and dashboard assets. Install the separate native wheel for its native aliases. Both include licensing notices. Low-level sources and the C header ship with the native wheel.

Build Linux from Linux, or use the pinned Rust Docker build from Windows:

```powershell
python scripts/build_native_hue.py --platform linux --test
```

The Linux wheel uses `linux_x86_64`, not manylinux. Check each target's libc compatibility. Local container checks do not establish hosted CI execution.

## Prepare a candidate

Run [tests](TESTING.md), review [the checklist](../RELEASE_CHECKLIST.md), scan public files and packages, and review third-party notices. Record failures and unexecuted checks. After source changes stop:

```powershell
python scripts/architecture.py --build
python scripts/architecture.py --check
python scripts/release_audit.py --check-docs
python scripts/release_audit.py --manifest
python scripts/scan_release.py
```

The manifest excludes Git metadata, environments, data caches and unselected artifacts. Ignore rules are a backup, not the selection policy. Repeat installation and checks in a fresh local checkout. Verify archive contents before distributing them. Keep raw security logs outside the public directory.

## Publication instructions

The approved identity is Brad Clonan with a GitHub noreply email. Only repository-local Git settings are used. No original history or remote is imported. After the maintainer creates an empty remote and authorizes publication, substitute its actual URL:

```powershell
Set-Location ../chromatic-oss
git status --short
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

`OWNER/REPOSITORY` is a placeholder. These commands have not been executed. Commercial exceptions require a separate agreement, and no pricing or sales backend exists.

## Recovery

Keep the untouched original as a recovery reference. Source changes invalidate source-bound receipts. Preserve failed runs and regenerate candidate evidence. Roll back a package by installing a previously verified wheel and checking its hash. There is no production database or hosted deployment to roll back. Save scientific measurements before removing generated directories.
