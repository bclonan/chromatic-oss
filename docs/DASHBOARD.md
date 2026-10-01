# Local research dashboard

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The dashboard reads measured experiment files and runs small codec inspections through the installed Python implementation. It needs no JavaScript build step and makes no external network requests.

From the project directory, start it with:

```powershell
python scripts/dashboard.py
```

Open [the local dashboard](http://127.0.0.1:8765). Stop the server with Ctrl+C in its terminal. To use another port or a saved run:

```powershell
python scripts/dashboard.py --port 8766 --results results/your-run.json
```

The installed package also supports `python -m dashboard.server`. The HTML, CSS, and JavaScript ship as package data.

## Recorded evidence

`results/latest.json` is the default input. Refresh results rereads the file. Export run downloads the complete loaded JSON. Neither action launches training or changes an experiment.

The interface has eleven views:

| View | What it shows |
| --- | --- |
| Overview | Run counts, validation curves, environment, and recorded findings |
| Experiments | Searchable task and representation filters, metrics, artifact downloads |
| Tensor viewer | Numerical tensor, actual display RGB, reconstruction errors, individual tokens |
| Color space | RGB, HSV, and circular hue projections of encoded tensors |
| Codebook | Learned representatives, display colors, assignment counts, paged vocabulary |
| Training | Training and validation loss, same-task same-seed comparisons, execution record |
| Benchmarks | Recorded seed means, sample standard deviations, timing, encoding bytes |
| Ablations | Precision by distribution and error domain, operator records, robustness records |
| Architecture | Implementation responsibilities, computation boundaries, saved configuration |
| Research conclusions | Findings saved by the runner and the limits of those findings |
| Cube Lab | Complete multidimensional projection, hidden-axis slices, integer turns, spectral erasure and reconstruction |

Charts have numerical tables. Tensor images support mouse inspection and keyboard inspection. Tab to an image, then use the arrow keys. Home and End select the first and last value. A visible frame marks the selected cell. The inspector reports original and decoded values, absolute error, RGB, calculated HSV, token identity, and frequency. The complete tensor also appears in a table.

Tensor snapshots with an experiment ID are labeled diagnostic encodings. Their inspection codec may differ from the representation used during training. Code and metadata bytes, live parameter bytes, checkpoint bytes, and process memory measure different things. The training view preserves the runner's execution and storage descriptions so these distinctions remain inspectable.

Precision filters keep input distributions and error domains separate. Linear error and circular hue error answer different questions. A single recorded seed has no measured seed-to-seed standard deviation.

The codebook view renders 128 entries per page. Previous codes, Next codes, and Inspect code support vocabularies with 65,536 entries without creating hundreds of thousands of DOM nodes.

## Live tensor inspection

Use Encode your own tensor in the tensor viewer. Enter a JSON array of numerical values and a matching shape such as `2, 2`. Choose a codec, then press Encode with Python codec.

The default codec is RGBD. Its table displays actual four-byte tokens, including D, the FP32 exponent. The RGB preview is the first three bytes. Values such as 1.0 and 2.0 share that preview but have different D bytes. Keep D when interpreting or exporting those values. This inspector creates real codec bytes without retaining an original floating-value cache in the stored tensor.

Codebook labels format their actual RGB bytes as hexadecimal `COLOR_RRGGBB`. Lookup IDs appear separately. A display label does not turn a palette ID into a lossless RGBD value. Learned RGBD uses a residual-table ID in D, which differs from the default codec's exponent.

The server uses the actual `ChromaTensor.from_tensor`, `codec.encode`, `codec.decode`, and `codec.colors` paths. It supports RGB, RGBD, HSV, circular hue, learned codebooks, learned RGBD modifiers, hierarchical residual codes, uniform INT8, uniform INT4, and 24-bit fixed-point.

Range-based codecs use the submitted minimum and maximum. Constant arrays get an expanded finite range. Learned vocabularies fit on the submitted array with seed 0, at most 64 primary representatives, and explicit fit-on-input metadata. These small inspections do not become benchmark results. Hue uses a period of 2π by default and reports circular reconstruction error.

Inputs must contain 1 to 4,096 finite numbers. Shapes have at most eight positive dimensions and must match the value count. The working tensor uses float32, and the inspector rejects magnitudes above 1e30. A JSON request body cannot exceed 262,144 bytes. Invalid requests produce a visible error and retain the input for correction.

## Live cube inspection

Cube Lab creates a periodic or seeded random tensor with three through six dimensions. Recorded tensor snapshots can also supply values if they contain enough cells. Source labels identify which path produced the cube. The default view projects every hidden-axis slice as related translucent cubes. Choose three display axes, select hidden coordinates, or orbit with the pointer and arrow keys. The numerical cell table provides the same codes and values without the canvas.

Turn controls choose a plane and a layer in a third axis. The server permutes integer codes and persistent cell identities together. Undo restores the previous complete state. Reset recreates the original tensor. Export state saves the current bounded JSON state. No turn launches model training.

The holographic view displays the actual quantized Fourier field. Phase sets hue and magnitude sets brightness. Coverage removes a reproducible set of coefficients. Decoded spectral reconstruction displays the inverse transform and its measured error. Full coverage still has phase and magnitude quantization error. Erasure loses information. A 3D projection cannot show additional axes without overlap or separate copies.

The cube endpoint accepts at most 1 MiB of JSON and 4,096 cells. The API supports side lengths two through eight within that cell limit. The UI selects side lengths two through four. Imported state must contain valid byte codes, unique identities, bounded codec parameters, and a matching shape. Canvas rendering has orbit and inspection keys, labeled controls, and a numerical table. Human screen-reader review remains unverified.

## HTTP endpoints

| Method and path | Behavior |
| --- | --- |
| `GET /api/health` | Reports service health and whether the result file exists |
| `GET /api/results` | Returns the saved schema version 1 run |
| `POST /api/tensor` | Encodes a bounded JSON input through the Python codecs |
| `POST /api/cube` | Creates, inspects, turns, or decodes a bounded multidimensional cube |
| `GET /api/cube-results` | Returns the independent saved cube experiment record |
| `GET /api/artifact?path=...` | Reads an allowed file beneath the results or project artifacts directory |

Example inspector request:

```json
{"values": [-0.5, 0, 0.5, 1], "shape": [2, 2], "codec": "rgb"}
```

The response contains the input, decoded values, integer codes, compound tokens, RGB and HSV values, frequencies, errors, codec configuration, and measured MAE and maximum error. Tokens retain all code channels, including hierarchical depth and learned RGBD residual modifiers.

Results follow this top-level contract:

```json
{
  "schema_version": 1,
  "run_id": "recorded-run-id",
  "created_at": "ISO 8601 timestamp",
  "status": "complete",
  "environment": {},
  "config": {},
  "experiments": [],
  "precision": [],
  "operators": [],
  "tensor_samples": [],
  "codebooks": [],
  "conclusions": []
}
```

Optional `robustness` rows appear in Ablations. Optional `instrumentation` metadata appears beside the saved configuration. Experiment records contain task, representation, seed, metrics, curves, execution, and artifact path. Tensor samples contain name, shape, values, decoded, RGB, tokens, and either a codec name or codec configuration. Codebooks contain numerical values, RGB colors, and measured frequencies. Missing sections show explicit empty states. A missing or unreadable result file never produces invented measurements.

## Observed installed dashboard

The original installed wheel displayed exact RGBD rows for `[1, 2, -1, 1.5]`, with zero measured MAE. Selecting 2 showed `[0,0,0,128]` and D equal to 128. The [preserved screenshot](../artifacts/dashboard/rgbd-current.jpg) shows that historical inspection. Its companion observations are omitted. Historical cube checks covered turn and undo, 729 cells in six dimensions, a 27-cell hidden slice and spectral erasure. The [archived acceptance report](history/results/acceptance.json) records the original run. These observations do not certify the candidate. Current browser evidence belongs in the release audit.

## Local hosting and troubleshooting

The server binds to `127.0.0.1` by default. It has no authentication and is intended for a trusted local machine. Keep that default for normal use. Binding to another address exposes the experiment files and codec endpoint to that network.

Static files come from a fixed allowlist. Artifact requests resolve paths and check that they remain beneath the results directory or the project artifacts directory. The permitted extensions are `.json`, `.csv`, `.txt`, `.md`, `.ctensor`, `.ctcheckpoint`, `.npz`, `.pt`, `.png`, `.jpg`, and `.jpeg`. Cube Lab exports its current state as JSON; the artifact endpoint does not download `.ccube` or `.hcube` files.

Every request must supply one Host header matching an allowed hostname and the server's actual port. POST requests with an Origin must match that HTTP authority. Requests without an Origin remain available to local command-line clients. Tensor and cube POST requests require JSON. The server sends a restrictive content security policy and disables content-type sniffing. It never executes artifact contents, launches subprocesses, starts training, or writes tensor inputs to disk. Public hosting still needs authentication, TLS, and separate deployment evidence.

Rejected POST requests discard a declared body only when it fits the one-MiB cleanup limit. The server reads in chunks with a one-second deadline and never parses that body. This lets Windows clients receive the HTTP denial before the connection closes. Oversized, invalid, or unfinished bodies do not create an unbounded cleanup wait. These limits govern rejected requests; accepted tensor inputs retain their smaller inspector budget.

If the page says no recorded run, execute `python scripts/run_benchmarks.py`, then refresh results. If a codec inspection says dependencies are missing, install the project in the same Python environment that runs the server. If port 8765 is occupied, pass another port. Reload the page after editing static files. Browser creation dates display in America/New_York; the exported JSON keeps the original timestamp.

Validation uses real codec round trips, bounded-input failures, result-file errors, HTTP content types, Host and Origin rejection, and path-traversal rejection in `tests/test_dashboard.py`. Browser verification exercises the ten result views and Cube Lab, loss charts, actual tensor encoding, arrow-key inspection, large codebook navigation, precision filters, and a narrow-screen layout. Current-source receipts determine which checks have passed. Screen-reader testing with a human operator remains a separate accessibility check.
