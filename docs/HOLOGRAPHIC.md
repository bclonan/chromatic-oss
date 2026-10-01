# Distributed Fourier cube

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

`HolographicCube` gives the multidimensional cube a global encoding. Its input is a finite real tensor with named axes. It applies an orthonormal N-dimensional Fourier transform across every axis, then stores each complex coefficient as a phase byte and two magnitude bytes. A spatial impulse changes every frequency coefficient. This is the precise sense in which this representation distributes information.

The code does not implement an optical hologram. It does not prove that a fragment contains enough information to recover an arbitrary original tensor. It introduces no claim of new Fourier mathematics, holographic reduced representations, or a machine learning advantage.

## Encoding and precision

The forward transform is `torch.fft.fftn(x.to(torch.float64), norm="ortho")`. A scalar input uses a complex128 identity transform. Empty inputs, complex inputs, NaN, infinity, and overflow are errors. The encoder retains no original floating tensor and no floating coefficient array.

Each coefficient stores three uint8 codes:

| Byte | Meaning | Quantization |
| --- | --- | --- |
| 0 | Phase | 256 equally spaced positions around the circle |
| 1 | Magnitude high byte | Upper eight bits of the magnitude integer |
| 2 | Magnitude low byte | Lower eight bits of the magnitude integer |

`phase_bins` is 256. `magnitude_levels` is the maximum integer 65535, so magnitude has 65536 inclusive states. A separate metadata scalar, `amplitude_scale`, stores the largest unquantized coefficient magnitude. The encoder rounds magnitude divided by this scale times 65535. It rounds phase to the nearest circular bin. Zero magnitude has phase code zero.

The payload uses 24 bits per original real scalar. Metadata adds shape, axis names, one amplitude scale, transform settings, byte counts, and a checksum. The format retains the full complex Fourier grid. Real input has redundant Hermitian coefficient pairs. This format does not pack only the independent half of that grid.

The decoder reconstructs complex128 coefficients and applies `torch.fft.ifftn(..., norm="ortho")`. It returns the real component in float64 by default. Quantization and masks can break exact Hermitian symmetry. The real projection discards the remaining imaginary component. Encoding and decoding perform floating point arithmetic. Integer storage does not make this an integer arithmetic execution method.

The RGB image is a preview, not the coefficient payload. HSV hue is quantized phase, saturation is one, and value is the magnitude code divided by 65535. This displays weak coefficients as darker cells. Actual amplitude still requires the separate scale metadata.

## Use

```python
import torch
from chromatensor.holographic import HolographicCube, numeric_polar_reference

generator = torch.Generator().manual_seed(824)
x = torch.randn(4, 4, 4, 3, 2, dtype=torch.float64, generator=generator)
cube = HolographicCube.from_tensor(x, ["x", "y", "z", "channel", "time"])
reconstructed = cube.decode()
colors = cube.rgb

# False entries erase frequency coefficients. The seed is local to this call.
keep = cube.erasure_mask(0.25, seed=103)
partial = cube.decode(mask=keep)
damaged_cube = cube.mask(keep)
assert torch.equal(partial, damaged_cube.decode())

cube.save("field.hcube")
loaded = HolographicCube.load("field.hcube")
assert torch.equal(loaded.codes, cube.codes)

# Equal-bit numerical labels give exactly the same reconstructed values.
control = numeric_polar_reference(x, mask=keep)
assert torch.equal(control, partial)
```

`coefficients()` returns the decoded complex field. `phase` returns radians. `magnitudes` returns scaled Fourier amplitudes. `shape`, `ndim`, `axis_names`, `device`, `dtype`, `storage_nbytes`, and `numel()` describe the stored object. `to`, `cpu`, `cuda`, and `clone` preserve byte storage and metadata. `decode(dtype=torch.float32)` makes an explicit numerical conversion after the float64 inverse transform.

`mask` requires a boolean torch tensor with exactly the cube's shape. It does not broadcast. `erasure_mask` erases the nearest integer count of the requested fraction. Its CPU generator gives the same mask on CPU and CUDA and leaves the global random state unchanged. `erase` returns a new stored cube containing the masked codes. The methods preserve the original amplitude scale, even if the largest coefficient is erased.

`encode_numeric_polar(spectrum)` returns byte codes and a scale. `decode_numeric_polar(codes, scale)` reconstructs complex coefficients. These functions also support a batched spectrum generated with explicit FFT dimensions. Their scale covers the whole supplied spectrum. To match individually encoded cubes, encode each example separately or use the same shared scale in both experimental conditions.

## Missing coefficients

Each retained frequency coefficient contributes across the reconstructed tensor. Erasing a coefficient therefore changes cells throughout the reconstruction. A smooth signal may remain recognizable after some coefficients disappear, especially when its energy is concentrated in coefficients that remain. White noise has no such guarantee. Different input tensors can agree on every retained coefficient and disagree on the missing coefficients.

Zero filling estimates missing coefficients as zero. It does not infer them. No decoder can uniquely recover arbitrary missing information without assumptions or extra evidence. Any learned reconstruction experiment must report its training distribution, missing coefficient mask, sample counts, and error against the original values. Successful recovery of smooth training examples does not establish arbitrary fragment recovery.

`numeric_polar_reference` uses the same 24-bit phase and magnitude levels, normalization, scale, mask, and inverse transform. Its reconstruction is identical to the cube's reconstruction. Hue labels alone cannot improve that control's numerical accuracy.

## Safe serialization

`.hcube` files contain a versioned magic header, bounded JSON metadata, uint8 byte codes with optional zlib compression, and a SHA-256 checksum over metadata and payload. They contain no pickle data or executable objects. The loader validates dimension counts, positive dimensions, exact byte allocation, supported numerical settings, axis names, finite scale, decompression bounds, and checksums. Writes use an atomic file replacement. The checksum detects accidental corruption. It is not an authenticity signature against a sender who can rewrite the checksum.

`HolographicCube.load` and `load_hcube` accept `max_file_bytes`, `max_header_bytes`, and `max_payload_bytes`. Callers can lower their default ceilings before loading. [File formats and limits](FILE_FORMATS.md) records those defaults. Loading and Fourier decoding can retain several buffers at once, so a payload limit does not cap peak RAM or device allocation.

## Sources and verification

The implementation follows PyTorch's documented [N-dimensional FFT](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fftn.html) and [inverse FFT](https://docs.pytorch.org/docs/2.14/generated/torch.fft.ifftn.html) behavior. Both transforms use orthonormal normalization. PyTorch documents the Hermitian redundancy of real input.

Run the numerical and format checks with the project Python environment:

```powershell
python -m pytest tests/test_holographic.py -q
```

The checks cover six-dimensional transforms, error bounds for seeded inputs, exact zero reconstruction, constant fields, scalar inputs, global impulse mixing, phase wrap, numerical control equality, information loss under erasure, CPU and CUDA agreement, and rejection of invalid or corrupted files.
