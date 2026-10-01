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


def metadata(title, url, description=None):
    description = description or "Brad Clonan's experiments with color tensors, neural weights and RGB LED hardware. Read the guides, source code and measured evidence."
    title = html.escape(title + " | Chromatic", quote=True)
    description = html.escape(description, quote=True)
    canonical = html.escape(BASE + url, quote=True)
    return f'''<title>{title}</title>
<meta name="description" content="{description}">
<meta name="author" content="Brad Clonan">
<meta name="robots" content="index,follow,max-image-preview:large">
<meta name="theme-color" content="#f8f7f2">
<link rel="canonical" href="{canonical}">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Chromatic">
<meta property="og:locale" content="en_US">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="{BASE}/assets/chromatic-social.png?v=hardware-build">
<meta property="og:image:secure_url" content="{BASE}/assets/chromatic-social.png?v=hardware-build">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="What if weights looked like color? Brad Clonan's experimental color tensors and RGB LED hardware, built with what he has.">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="{BASE}/assets/chromatic-social.png?v=hardware-build">
<meta name="twitter:image:alt" content="Chromatic experimental color tensors and a numeric-to-color network diagram.">'''


def page(title, content, url="/", home=False):
    links = "".join(f'<a href="{url}">{label}</a>' for label, url in NAV)
    head = metadata(title, url, "RGB LEDs, some grit and a build with my son. Explore Brad Clonan's experiments with color, neural weights and light, with a native library and research evidence." if home else None)
    if home:
        head += f'''<meta property="og:video" content="{BASE}/assets/media/hardware-build.mp4">
<meta property="og:video:secure_url" content="{BASE}/assets/media/hardware-build.mp4">
<meta property="og:video:type" content="video/mp4">
<meta property="og:video:width" content="720">
<meta property="og:video:height" content="1280">'''
        video_data = {"@context": "https://schema.org", "@type": "VideoObject",
                      "name": "Building the next experiment with my son",
                      "description": "Brad Clonan and his son assemble experimental RGB LED hardware. A camera enclosure is planned to control ambient light and test reliable color readback. Physical neural computation is unproven.",
                      "thumbnailUrl": BASE + "/assets/media/hardware-build-poster.png",
                      "contentUrl": BASE + "/assets/media/hardware-build.mp4",
                      "uploadDate": "2026-10-01", "duration": "PT1M28.767S"}
        head += '<script type="application/ld+json">' + json.dumps(video_data).replace("<", "\\u003c") + '</script>'
    extra = '<link rel="stylesheet" href="/assets/home.css"><script type="module" src="/assets/home.js"></script>' if home else ''
    header_right = '<nav class="header-links" aria-label="Main"><a href="/catalog.html">Documentation</a><a href="/docs/architecture/index.html">Explorer ↗</a><a href="https://github.com/bclonan/chromatic-oss">GitHub ↗</a></nav>' if home else '<span class="edition">Documentation / 0.1.0 native</span>'
    sidebar = '' if home else f'<nav aria-label="Documentation">{links}<hr><a href="/downloads/chromatic-oss.zip">Download source ZIP</a><a href="/source/">Browse release files</a></nav>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{head}
<link rel="stylesheet" href="/site.css">{extra}</head><body class="{'home' if home else 'document'}">
<a class="skip" href="#content">Skip to content</a>
<header><a class="brand" href="/">CHROMATIC <span>color as information</span></a>{header_right}</header>
<div class="layout">{sidebar}
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
            if source.suffix == ".html":
                copied = target.read_text(encoding="utf-8")
                heading = "Architecture explorer" if source.name == "index.html" else "Architecture source reference"
                copied = re.sub(r"<title>.*?</title>", metadata(heading, "/" + name), copied, count=1)
                target.write_text(copied, encoding="utf-8")
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
        target.write_text(page(title, raw + rendered, "/" + output_name), encoding="utf-8")
        documents.append((title, output_name, name))

    for directory in sorted((OUT / "source").rglob("*"), reverse=True):
        if directory.is_dir() and not (directory / "index.html").exists():
            listing(directory)
    listing(OUT / "source")
    catalog = '<h1>All documents</h1><p>Guides, research reports, release records and third-party notices. Historical reports retain their original scope and source identities.</p><ul class="catalog">'
    catalog += "".join(f'<li><a href="/{url}">{html.escape(title)}</a><small>{name}</small></li>' for title, url, name in documents)
    (OUT / "catalog.html").write_text(page("All documents", catalog + "</ul>", "/catalog.html"), encoding="utf-8")
    home = (ROOT / "docs/site/home.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(page("What if weights looked like color?", home, home=True), encoding="utf-8")
    (OUT / "assets").mkdir()
    for name in ("home.css", "home.js", "chromatic-social.png", "favicon.svg"):
        if "docs/site/" + name not in selected:
            raise ValueError("Website asset is not in the release manifest: " + name)
        shutil.copyfile(ROOT / "docs/site" / name, OUT / "assets" / name)
    for name in sorted(selected):
        if name.startswith("docs/site/media/") and Path(name).suffix in {".mp4", ".png", ".vtt"}:
            destination = OUT / "assets/media" / Path(name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    urls = ["/", "/catalog.html", "/docs/architecture/index.html"] + ["/" + url for _, url, _ in documents]
    sitemap = '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + "".join("<url><loc>" + html.escape(BASE + url) + "</loc></url>" for url in urls) + "</urlset>"
    (OUT / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    (OUT / "robots.txt").write_text("User-agent: *\nAllow: /\nSitemap: " + BASE + "/sitemap.xml\n", encoding="utf-8")
    shutil.copyfile(ROOT / "docs/site/site.css", OUT / "site.css")
    (OUT / "404.html").write_text(page("Page not found", '<h1>Page not found</h1><p><a href="/catalog.html">Browse all documents</a>.</p>', "/404.html"), encoding="utf-8")
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
    (directory / "index.html").write_text(page("Release files", '<h1>Release files</h1><ul>' + "".join(items) + "</ul>", "/" + directory.relative_to(OUT).as_posix() + "/"), encoding="utf-8")


def check_links():
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in {"href", "src", "poster"} or not value:
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
