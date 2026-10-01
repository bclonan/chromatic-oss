"""Protect the reviewed selection boundary and exact archive contents."""
import json
from pathlib import Path
import zipfile

import pytest

from scripts.release_audit import MANIFEST, build_manifest, check_documentation, create_archive, select_files


def test_manifest_excludes_private_and_generated_material(tmp_path):
    files = {"README.md": "public", "chromatensor/tensor.py": "pass", "docs/guide.md": "public",
             "chromatensor/_native/windows-x86_64/library.dll": "private binary", ".env": "secret",
             ".git/config": "private remote", "data/customer.json": "private",
             "artifacts/local.log": "private", "native/hue_tensor/target/debug/output.txt": "private",
             ".venv/Lib/site-packages/cache.py": "private", "docs/private.pem": "private"}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    assert {path.relative_to(tmp_path).as_posix() for path in select_files(tmp_path)} == {"README.md", "chromatensor/tensor.py", "docs/guide.md"}


def test_archive_checks_selected_bytes_and_exact_members(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "README.md").write_text("source")
    manifest = root / MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(build_manifest(root)))
    destination = tmp_path / "release.zip"
    assert create_archive(destination, root)["members"] == 2
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"chromatic-oss/README.md", "chromatic-oss/" + MANIFEST}
    (root / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        create_archive(destination, root)


def test_doc_links_must_resolve_within_candidate(tmp_path):
    (tmp_path / "README.md").write_text("[bad](../private.md)\n[missing](absent.md)\n[web](https://example.invalid)")
    report = check_documentation(tmp_path)
    assert report["status"] == "FAIL"
    assert report["local_links_checked"] == 2
    assert len(report["errors"]) == 2


def test_native_notices_and_c_consumers_are_selected(tmp_path):
    for name in ("native/rust-notices/licenses/MIT", "native/rust-notices/manifest.json", "native/examples/consumer.c"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed source or attribution")
    assert len(select_files(tmp_path)) == 3
