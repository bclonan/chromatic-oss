# Third-party notices

The maintainer selected AGPL-3.0-only for this project's source and may offer separate paid exceptions. See [LICENSE](LICENSE) and the release licensing documentation for the exact grant. An exception requires a separate agreement with the relevant rights holder. The project license does not replace upstream licenses or dataset terms.

No third-party Python packages, dataset archives, pretrained third-party models, fonts, browser libraries, or Rust crates are vendored in the reviewed source release. The Rust implementation's Cargo lock contains only `hue_tensor`. Python dependencies install through pip and retain their upstream metadata and license files. Redistribution of an environment, container, binary, or wheel that bundles dependencies needs its own contents and notice review.

## Python dependencies

The direct dependencies are PyTorch, torchvision, NumPy, Matplotlib, and psutil. pytest supports tests and setuptools supports builds. Their installed distributions contain their own licenses and notices.

| Dependency | License family or installed license expression | Upstream source |
| --- | --- | --- |
| PyTorch | BSD licenses and bundled components under Apache-2.0, LLVM exception, Boost, and MIT | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | BSD | [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| NumPy | BSD-3-Clause with bundled 0BSD, MIT, Zlib, and CC0 material | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Matplotlib | Matplotlib license based on the Python Software Foundation license | [Matplotlib license](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| psutil | BSD-3-Clause | [psutil license](https://github.com/giampaolo/psutil/blob/master/LICENSE) |
| pytest | MIT | [pytest license](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| setuptools | MIT | [setuptools license](https://github.com/pypa/setuptools/blob/main/LICENSE) |

The license audit command records installed versions, license metadata, license classifiers, and upstream license-file names for every installed package named in the committed requirement files. Those observations apply to that environment. Lockfiles for other Python versions are dependency specifications, they are not proof that every upstream file in every platform wheel was inspected.

## Native binary notices

Native hue binaries link Rust standard-library code. [native/rust-notices](native/rust-notices/README.md) preserves the official Rust 1.96.0 copyright inventory and license texts with byte hashes. Native wheels bundle those files with the project license and copyright statement. The inventory covers the whole upstream toolchain, so it is broader than the components linked by this library. Its inclusion does not assert that every named crate, compiler component, or documentation asset is part of `hue_tensor`.

## Datasets

`experiments/datasets.py` downloads real datasets into the configured local data directory. Dataset files and saved input arrays are excluded from the public source manifest. Keep the original sources and terms when running experiments or redistributing any resulting data.

| Dataset | Attribution and release treatment |
| --- | --- |
| MNIST | Handwritten digits by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. The runner uses torchvision's MNIST loader. [Dataset documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html). No dataset archive or image arrays are bundled. |
| CIFAR-10 | Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cite Alex Krizhevsky's 2009 report, Learning Multiple Layers of Features from Tiny Images. The [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) requests this citation. That page does not supply a clear general redistribution license. No dataset archive or image arrays are bundled. |
| Tiny Shakespeare | The runner downloads `data/tinyshakespeare/input.txt` from Andrej Karpathy's [char-rnn repository](https://github.com/karpathy/char-rnn). Its README declares MIT for that repository. No corpus file is bundled. Preserve the dataset source and corpus hash in any research report. |

The source contains analytic scalar, periodic-field, and seeded synthetic retrieval tasks. Reports identify them by their own experiment names. They do not stand in for real MNIST, CIFAR-10, or language data.

## Generated assets and research references

The dashboard CSS and JavaScript are project source. Its tensor colors and cube displays come from numeric inputs. Project plots, architecture data, and the explorer come from project code. Historical research evidence retains dataset names, sample counts, seeds, measured storage, execution precision, and original hashes. Sanitized records are historical observations and do not establish current-source verification.

Vector quantization, straight-through gradients, and Fourier transforms are established techniques. [THEORY.md](THEORY.md) links the research references used by the implementation. The project does not claim ownership of those underlying methods or a novel accuracy advantage from color labels.

## Documentation build dependency

Python-Markdown 3.10.2 converts the selected Markdown documents to HTML. It is a build dependency under the BSD-3-Clause license and is not bundled in the website or source ZIP. Its distribution retains its upstream license and metadata.
