# Rust toolchain notices

This directory preserves the installed official Rust 1.96.0 copyright inventory and its twelve license texts. [manifest.json](manifest.json) records exact byte counts, SHA-256 hashes, the compiler identity, and the source policy. The files retain their original contents and attribution.

The native hue library links code from Rust's standard library. Native wheels include these notices alongside the project's AGPL license, project copyright statement, and third-party notices. The full Rust inventory also describes compiler components, documentation, and tools. Its inclusion does not mean that every listed component is linked into `hue_tensor`.

Update these files from the official compiler distribution when changing the supported Rust toolchain. Copy `share/doc/rust/COPYRIGHT.html` and `share/doc/rust/licenses/*.txt`, record the compiler's `rustc --version`, and regenerate the manifest hashes. Review the copied bytes for private data and preserve legitimate upstream copyright, authors, contact addresses, and license wording. The native builder checks the actual build compiler against this notice inventory.

See the [Rust license policy](https://rust-lang.org/policies/licenses/). The public URLs for the `home` crate inside the original copyright inventory are upstream package references. They are not personal filesystem paths. The privacy scanner has a narrow exception for those versioned public crate URLs and still rejects ordinary home-directory paths.
