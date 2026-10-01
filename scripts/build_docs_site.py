"""Build a static documentation site from hash-verified public release files."""
from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import posixpath
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

import markdown

from release_audit import ROOT, MANIFEST, create_archive
from scan_release import scan_tree

OUT = ROOT / "dist/docs"
BASE = "https://chromatic-hue.netlify.app"
NAV = [("Start", "/README.html"), ("User guide", "/docs/USER_GUIDE.html"),
       ("Native API", "/docs/NATIVE_HUE.html"), ("Development", "/docs/DEVELOPER_GUIDE.html"),
       ("Architecture explorer", "/docs/architecture/index.html"), ("All documents", "/catalog.html")]


def page(title, content):
    links = "".join(f'<a href="{url}">{label}</a>' for label, url in NAV)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} | Chromatic hue tensors</title>
<meta name="description" content="Rust RGBD tensors with exact modular hue arithmetic. Guides, source code and research evidence.">
<link rel="stylesheet" href="/site.css"></head><body>
<a class="skip" href="#content">Skip to content</a>
<header><a class="brand" href="/">CHROMATIC <span>hue tensors</span></a><span class="edition">Documentation / 0.1.0 native</span></header>
<div class="layout"><nav aria-label="Documentation">{links}<hr><a href="/downloads/chromatic-oss.zip">Download source ZIP</a><a href="/source/">Browse release files</a></nav>
<main id="content" tabindex="-1">{content}</main></div>
<footer>Copyright 2026 Brad Clonan. <a href="/source/LICENSE">AGPL-3.0-only</a> · <a href="/THIRD_PARTY_NOTICES.html">Third-party notices</a> · <a href="/docs/HOSTING.html">Hosting and rebuild instructions</a></footer>
</body></html>'''


def main():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    selected = {row["path"] for row in manifest["files"]} | {MANIFEST}
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT) or path.is_symlink():
            raise ValueError("Unsafe release path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Refresh release manifest before building: " + row["path"])
    scan = scan_tree(ROOT, ROOT / MANIFEST)
    if scan["status"] != "PASS":
        raise ValueError("Release privacy scan failed")
    # Only this fixed generated directory may be replaced.
    if OUT.resolve() != ROOT / "dist" / "docs" or OUT.is_symlink():
        raise ValueError("Unsafe output directory")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    documents = []
    for name in sorted(selected):
        source = ROOT / name
        target = OUT / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if name.startswith("docs/architecture/"):
            target = OUT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if source.name == "index.html":
                target.write_text(source.read_text(encoding="utf-8").replace('../../ARCHITECTURE.md', '/ARCHITECTURE.html'), encoding="utf-8")
            elif source.name == "source.js":
                target.write_text(source.read_text(encoding="utf-8").replace('"../../"', '"/source/"').replace("'../../'", "'/source/'"), encoding="utf-8")
        if source.suffix != ".md":
            continue
        text = source.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, re.M)
        title = title_match[1].strip() if title_match else source.stem.replace("_", " ")
        rendered = markdown.markdown(text, extensions=["fenced_code", "tables", "toc", "sane_lists"])

        def link(match):
            attr, url = match.groups()
            parsed = urlsplit(html.unescape(url))
            if parsed.scheme or parsed.netloc or not parsed.path:
                return match[0]
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(parsed.path)))
            if resolved.endswith(".md") and resolved in selected:
                result = "/" + resolved[:-3] + ".html"
            elif resolved.startswith("docs/architecture/"):
                result = "/" + resolved
            else:
                result = "/source/" + resolved
                if parsed.path.endswith("/"):
                    result += "/"
            if parsed.fragment:
                result += "#" + parsed.fragment
            return f'{attr}="{html.escape(result, quote=True)}"'

        rendered = re.sub(r'(href|src)="([^"]+)"', link, rendered)
        raw = f'<p class="source-link"><a href="/source/{name}">Read Markdown source</a></p>'
        output_name = name[:-3] + ".html"
        target = OUT / output_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page(title, raw + rendered), encoding="utf-8")
        documents.append((title, output_name, name))

    for directory in sorted((OUT / "source").rglob("*"), reverse=True):
        if directory.is_dir() and not (directory / "index.html").exists():
            listing(directory)
    listing(OUT / "source")
    catalog = '<h1>All documents</h1><p>Guides, research reports, release records and third-party notices. Historical reports retain their original scope and source identities.</p><ul class="catalog">'
    catalog += "".join(f'<li><a href="/{url}">{html.escape(title)}</a><small>{name}</small></li>' for title, url, name in documents)
    (OUT / "catalog.html").write_text(page("All documents", catalog + "</ul>"), encoding="utf-8")
    home = '''<p class="eyebrow">Rust library / Python / C / CLI</p><h1>Color tokens.<br>Exact hue arithmetic.</h1>
<p class="lead">A tensor library that stores four-byte RGBD tokens and computes in the finite ring Z/391680Z.</p>
<div class="actions"><a href="/docs/USER_GUIDE.html">Start using the library</a><a href="/docs/architecture/index.html">Explore the architecture</a></div>
<div class="facts"><div><strong>4 bytes</strong><span>per tensor element</span></div><div><strong>391,680</strong><span>discrete phase positions</span></div><div><strong>CPU</strong><span>native integer operations</span></div></div>
<h2>What it does</h2><p>Encode canonical colors, add and multiply hue phases, contract matrices, reshape tensors and save bounded, checksummed files. Python and C call the same Rust implementation. The native wheel has no third-party runtime dependencies.</p>
<pre><code>from chromatic_hue import HueEncoder

with HueEncoder.encode_phases([[1, 2], [3, 4]]) as a:
    with a @ a as result:
        print(result.phases())  # (7, 10, 15, 22)</code></pre>
<h2>Read the evidence</h2><p>Each element occupies 32 physical bits. Recorded native operators were slower than equal-storage NumPy controls. No compression, learning or hardware-speed advantage has been demonstrated.</p>
<p>The legacy PyTorch experiments use floating-point training and execution. Historical receipts describe their original source. They do not certify later changes.</p>
<div class="actions"><a href="/docs/RELEASE_AUDIT.html">Release audit</a><a href="/LIMITATIONS.html">Limitations</a><a href="/catalog.html">All documents</a></div>
<h2>Run it locally</h2><p>The native library, CLI and research dashboard run on your machine. This website hosts documentation, the architecture explorer and reviewed release files. It does not run the Python dashboard or numerical experiments.</p>
<p><a href="/downloads/chromatic-oss.zip">Download the approved source release</a>, then follow the <a href="/docs/DEVELOPER_GUIDE.html">setup instructions</a>. The ZIP includes the license, copyright and third-party notices.</p>'''
    (OUT / "index.html").write_text(page("Color tokens. Exact hue arithmetic.", home), encoding="utf-8")
    shutil.copyfile(ROOT / "docs/site/site.css", OUT / "site.css")
    (OUT / "404.html").write_text(page("Page not found", '<h1>Page not found</h1><p><a href="/catalog.html">Browse all documents</a>.</p>'), encoding="utf-8")
    (OUT / "downloads").mkdir()
    with tempfile.TemporaryDirectory(prefix="chromatic-docs-") as temp:
        archive = Path(temp) / "chromatic-oss.zip"
        archive_info = create_archive(archive)
        shutil.copyfile(archive, OUT / "downloads/chromatic-oss.zip")
    check_links()
    print(json.dumps({"status": "PASS", "documents": len(documents), "release_files": len(selected), "archive": archive_info}))


def listing(directory):
    items = []
    for item in sorted(directory.iterdir()):
        url = "/" + item.relative_to(OUT).as_posix() + ("/" if item.is_dir() else "")
        items.append(f'<li><a href="{html.escape(url)}">{html.escape(item.name)}</a></li>')
    (directory / "index.html").write_text(page("Release files", '<h1>Release files</h1><ul>' + "".join(items) + "</ul>"), encoding="utf-8")


def check_links():
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in {"href", "src"} or not value:
                    continue
                url = urlsplit(value)
                if url.scheme or url.netloc or not url.path:
                    continue
                target = OUT / unquote(url.path).lstrip("/") if url.path.startswith("/") else current.parent / unquote(url.path)
                if not target.exists():
                    errors.append((current.relative_to(OUT).as_posix(), value))
    errors = []
    for current in OUT.rglob("*.html"):
        if "source" in current.relative_to(OUT).parts:
            continue
        Links().feed(current.read_text(encoding="utf-8"))
    if errors:
        raise ValueError("Broken website links: " + repr(errors))


if __name__ == "__main__":
    main()
