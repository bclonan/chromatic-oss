"""Negative controls for redacted release scanning and archive privacy checks."""
import importlib.util
import io
import json
from pathlib import Path
import zipfile

spec = importlib.util.spec_from_file_location("release_scan", Path(__file__).resolve().parents[1] / "scripts" / "scan_release.py")
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


def zipped(name, content):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, content)
    return stream.getvalue()


def test_nested_archive_finds_credential_and_does_not_report_value():
    token = "gh" + "p_" + "example" * 6
    findings, limits = scanner.scan_bytes(zipped("inner.zip", zipped("config.txt", token)), "release.zip")
    assert not limits
    assert any(f["kind"] == "hosted_token" and f["path"].endswith("!inner.zip!config.txt") for f in findings)
    assert token not in json.dumps(findings)


def test_manifest_outside_root_fails_without_reading(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "../outside.txt"}]}))
    report = scanner.scan_tree(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert report["files_scanned"] == 0
    assert report["finding_counts"] == {"invalid_manifest_path": 1}


def test_private_configuration_is_rejected_even_without_token(tmp_path):
    (tmp_path / ".env").write_text("VALUE=123")
    report = scanner.scan_tree(tmp_path)
    assert report["status"] == "FAIL"
    assert report["finding_counts"] == {"private_configuration_file": 1}


def test_public_noreply_and_fake_email_are_preserved(tmp_path):
    (tmp_path / "NOTICE").write_text("Brad Clonan <bclonan@users.noreply.github.com>\nSynthetic identity <evidence@example.invalid>\n")
    assert scanner.scan_tree(tmp_path)["status"] == "PASS"


def test_private_user_path_is_redacted_from_location():
    path = "C:" + "\\" + "Users" + "\\" + "private-test-user" + "\\report.txt"
    findings, limits = scanner.scan_bytes(path.encode(), path)
    assert not limits
    assert findings[0]["kind"] == "private_user_path"
    assert "private-test-user" not in json.dumps(findings)


def test_public_rust_home_crate_url_is_preserved_and_private_paths_still_fail():
    public = "https://crates.io/crates/home/0.5.11"
    findings, limits = scanner.scan_bytes(public.encode(), "COPYRIGHT.html")
    assert findings == [] and limits == []
    private = "/" + "home" + "/" + "private-test-user" + "/data"
    findings, _ = scanner.scan_bytes(private.encode(), "config.txt")
    assert findings[0]["kind"] == "private_user_path"


def test_archive_traversal_is_rejected():
    findings, limits = scanner.scan_bytes(zipped("../outside.txt", "harmless"), "release.zip")
    assert not limits
    assert findings == [{"kind": "unsafe_archive_member", "path": "release.zip!../outside.txt", "line": 0}]


def test_archive_private_configuration_is_rejected_even_without_token():
    findings, limits = scanner.scan_bytes(zipped("config/.env", "VALUE=123"), "release.zip")
    assert not limits
    assert findings == [{"kind": "private_configuration_file", "path": "release.zip!config/.env", "line": 0}]


def test_archive_symlink_requires_explicit_review():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("linked.txt")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "safe-target.txt")
    findings, limits = scanner.scan_bytes(stream.getvalue(), "release.zip")
    assert not limits
    assert findings == [{"kind": "symlink_requires_review", "path": "release.zip!linked.txt", "line": 0}]


def test_unreadable_nested_archive_is_never_a_passing_scan(monkeypatch):
    monkeypatch.setattr(scanner, "MAX_DEPTH", 1)
    _, limits = scanner.scan_bytes(zipped("inner.zip", zipped("file.txt", "text")), "release.zip")
    assert limits == [{"path": "release.zip!inner.zip", "reason": "archive depth limit"}]


def test_missing_selected_file_is_not_silently_skipped(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": ["missing.txt"]}))
    report = scanner.scan_tree(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert report["limits"] == [{"path": "missing.txt", "reason": "missing selected file"}]
