# Using chromatic tensors

The native package stores four RGBD bytes per element and computes in the finite ring `Z/391680Z`. It is useful for experimenting with discrete hue composition, inspecting exact color storage and testing language bindings. Its matrix product uses that ring. It does not reproduce real-number matrix multiplication or physical color mixing.

Build and install the native wheel using the [release guide](RELEASE_GUIDE.md). Then run:

```python
from chromatic_hue import Tensor, HueEncoder, HueDecoder

with HueEncoder.encode_phases([[1, 2], [3, 4]]) as a:
    with HueEncoder.encode_phases([[5, 6], [7, 8]]) as b:
        with a @ b as answer:
            assert HueDecoder.decode_phases(answer) == (19, 22, 43, 50)
            assert answer.rgbd_bytes[:4] == bytes([255, 0, 0, 19])
            assert answer.storage_nbytes == 16
            assert answer.save("answer.huetensor") == 88

with Tensor.load("answer.huetensor", max_bytes=1024) as restored:
    assert restored.phases() == (19, 22, 43, 50)
```

The encoder projects ordinary RGB onto a saturated hue wheel and rejects gray. It loses brightness and saturation. Canonical RGBD input preserves all four bytes. `HueDecoder` returns bytes or integer phase ticks. Context managers close native handles. A closed tensor rejects further operations. See [the native API guide](NATIVE_HUE.md) for broadcasting, shape changes, Rust, C and CLI examples.

Files contain shape metadata, RGBD bytes and a SHA-256 checksum. Loading applies byte and element budgets and validates the shape and colors. A checksum detects corruption; it does not establish who created a file. Never raise file budgets for untrusted input without reviewing the memory cost.

## Earlier numerical research

Install the research dependencies using the [developer guide](DEVELOPER_GUIDE.md). The following run creates seeded scalar, periodic, precision, operator and attention measurements without downloading real datasets:

```powershell
.venv/Scripts/python.exe scripts/run_benchmarks.py --profile smoke --device cpu --seeds 7 --tasks scalar periodic precision operators attention --output results/local-smoke.json
.venv/Scripts/python.exe scripts/dashboard.py --results results/local-smoke.json
```

Open `http://127.0.0.1:8765`. The overview reads the specified result file. The tensor viewer calls the real codecs with bounded numeric inputs. Cube Lab creates actual integer-backed geometry and a software Fourier field. Turn a layer, undo it, select a hidden-axis slice, or erase coefficients and inspect reconstruction. These are legacy numerical APIs. Their codecs and FFT work use floating-point PyTorch where documented.

```powershell
.venv/Scripts/python.exe scripts/run_cube_experiments.py --seeds 7 --epochs 2 --output results/cube.json
```

This small cube run trains reconstruction models on the stated synthetic generator. It tests execution, not a general learning advantage. MNIST and CIFAR tasks download their actual datasets into `data/`; they do not silently replace them with synthetic examples. The language experiment downloads Tiny Shakespeare and samples training and validation windows from its 90/10 split. Read [measurement](MEASUREMENT.md) and [the experiment matrix](../EXPERIMENT_MATRIX.md) before interpreting historical results.

Historical measurements retain their seeds, sample counts and precision descriptions. Public sanitization removes machine-specific paths and omits raw dataset-derived snapshots. The source includes the regeneration commands. Historical source hashes do not certify this release candidate.

## Errors and cleanup

Invalid shapes, noncanonical colors, negative budgets, corrupt files and closed handles raise errors. The dashboard rejects wrong hosts and cross-origin requests, oversized bodies and unknown codecs. Stop its local server with Ctrl+C. Delete generated environments, `data/`, `build/` and unneeded local results only after preserving measurements you want to keep. No remote retention or account deletion exists because this project has no accounts or hosted data service.
