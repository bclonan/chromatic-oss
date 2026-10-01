# Public documentation hosting

The documentation site is at [chromatic-hue.netlify.app](https://chromatic-hue.netlify.app). It includes the user guide, native API, development and testing instructions, architecture explorer, research documents, historical receipts, reviewed figures and a source ZIP.

The site is static. The Python research dashboard and native numerical execution run locally. Follow [the developer guide](DEVELOPER_GUIDE.md) to run them.

## Rebuild

Install the documentation build dependency, regenerate the architecture data and release manifest, then build:

```sh
python -m pip install -r requirements/docs.txt
python scripts/architecture.py --build
python scripts/architecture.py --check
python scripts/release_audit.py --check-docs --manifest
python scripts/scan_release.py --manifest docs/release/manifest.json
python scripts/build_docs_site.py
```

The builder verifies every manifest hash and runs the privacy scanner before writing `dist/docs`. It copies only manifest-selected files, renders Markdown as HTML, checks generated document links and creates a source ZIP with LICENSE, COPYRIGHT and THIRD_PARTY_NOTICES.md. The source ZIP includes the site builder. Raw logs, credentials, datasets, original Git history and unselected build outputs remain excluded.

The renderer uses Python-Markdown 3.10.2 as a build dependency. It does not ship the renderer or add a browser dependency. Original Markdown remains available under `/source/`. The architecture explorer retains its own static files and source snapshots.

## Deploy

Use an authenticated Netlify CLI with the existing project:

```sh
npx netlify deploy --site 7823a2ee-6732-4208-8bde-275ca4d8c4e6 --dir dist/docs --no-build --prod
```

`netlify.toml` also defines the build command and output folder for a future Git-connected build. This release uses a direct CLI deployment. It does not establish automatic GitHub deployment.

The GitHub repository is maintained separately. Publishing these selected files does not change its visibility or publish its original history. Redeploy a previously checked build to roll back website content.

## Visual experiment page

The main page includes a plain JavaScript and SVG network illustration, a native hue encoding illustration, links to all documentation and the maintainer's public social profiles. Numeric and color views run the same fixed-weight scalar network. The native hue illustration follows the Rust byte mapping but does not execute the Rust library in the browser. Neither illustration trains a model.

Run `node scripts/check_home_animation.js` to check all 391,680 phase round trips, boundary bytes, invalid inputs and reference network outputs. The renderer reads `docs/site/home.html`, `home.css` and `home.js`. It honors reduced motion and provides pause and step controls. Mobile visitors can scroll the network diagram horizontally.

Rendered documents and architecture pages include canonical URLs, Open Graph tags and X large-image card tags. The site publishes a sitemap, robots file, favicon and a 1200 by 630 PNG share card. To regenerate the card, install Pillow 12.3.0 and run `python scripts/build_social_card.py` with Arial or DejaVu Sans available locally. The website ships the PNG, not Pillow or font binaries. Social platforms may cache older previews; their rendered cards were not verified during this deployment.
