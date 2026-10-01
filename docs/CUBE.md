# Cube geometry

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

ChromaCube stores an N-dimensional ChromaTensor and an integer ID for every scalar cell. The cell carries all of its codec channels when it moves. IDs identify the original row-major positions, so the system can track a cell even when several cells have the same color or value.

```python
import torch
from chromatensor.cube import ChromaCube

cube = ChromaCube.from_tensor(
    torch.linspace(-1, 1, 3 ** 6).reshape(3, 3, 3, 3, 3, 3),
    axis_names=["x", "y", "z", "depth", "hue", "time"],
)
turned = cube.twist(axes=("x", "y"), layer_axis="z", layer_index=1)
restored = turned.inverse_twist(axes=("x", "y"), layer_axis="z", layer_index=1)
assert torch.equal(restored.tensor.codes, cube.tensor.codes)
assert torch.equal(restored.cell_ids, cube.cell_ids)
```

Axis names describe tensor coordinates. Naming an axis "hue" does not change the codec or make that coordinate a physical light dimension.

## What a turn does

A positive turn rotates the ordered axis plane counterclockwise, following `torch.rot90`. In a 3 by 3 plane with axes x then y, new cell `x, y` receives old cell `y, 2 - x`. Reversing the plane order reverses the direction. `turns` counts quarter turns and accepts negative integers.

A layer turn fixes one coordinate on a third axis. In a six-dimensional cube, it repeats the plane rotation for every combination of the other three coordinates. Set `layer_axis=None` to rotate every cell in the selected plane. An odd number of quarter turns requires equal plane lengths. A half turn also works on a rectangular plane. The cube needs at least three dimensions and every dimension must contain at least one cell.

`twist` returns a new cube. It keeps the original codes and IDs unchanged. Four quarter turns restore every code and ID. Applying `inverse_twist` with the same arguments undoes the corresponding turn. To undo a sequence, apply its inverse turns in reverse order.

Turns permute the existing code bytes and integer IDs with PyTorch operations. Rotating byte blocks preserves unsigned code containers even where PyTorch does not support rotation on that integer dtype. Turns do not decode values, interpolate colors, train a model, or evaluate a neural network. A rotation changes where a value resides. It does not change the stored value. The ID tensor uses 8 bytes per cell in addition to the codec's code storage.

## Viewing three dimensions

```python
view = turned.slice3d(
    axes=("time", "z", "x"),
    fixed={"y": 2, "depth": 1, "hue": 0},
)
tensor_slice = view["tensor"]
cell_ids = view["cell_ids"]
```

The returned `tensor` is a ChromaTensor. `cell_ids` has the same logical shape. Their dimensions follow the exact order in `axes`, so the example returns time, z, then x. `axis_names` identifies that order, `axes` supplies normalized integer axis indices, and `fixed` records the coordinates used for omitted dimensions. Unspecified omitted coordinates default to zero. Fixed coordinates cannot name a displayed axis.

A view can show only three dimensions at once. Changing the displayed axes and the fixed coordinates reveals other cells from the same N-dimensional cube. A holographic drawing is a display treatment for these slices. It does not create a physical hologram or display all independent dimensions at once.

## Devices and files

`cube.to("cuda")` moves both the chromatic codes and the cell IDs to CUDA. `cube.decode()` explicitly converts the stored codes to numerical values. Geometry operations preserve the storage dtype.

```python
turned.save("artifacts/six_dimensions.ccube")
loaded = ChromaCube.load("artifacts/six_dimensions.ccube", device="cpu")
```

The `.ccube` file is a ZIP container with a checked `.ctensor` payload, little-endian int64 IDs, JSON metadata, and SHA-256 checksums. Loading checks the entries, byte counts, dimensions, axis names, and complete ID permutation. It does not load pickle or execute Python objects. Writes replace the destination atomically. Checksums detect accidental corruption. They do not authenticate the author of a file.

`ChromaCube.load` accepts `max_file_bytes`, `max_header_bytes`, `max_payload_bytes`, and `max_expanded_bytes`. The payload budget includes both codes and int64 identities. The archive has exactly four members. [File formats and limits](FILE_FORMATS.md) gives the defaults and smaller-budget examples. Those limits do not cap peak process RAM or device copies.
