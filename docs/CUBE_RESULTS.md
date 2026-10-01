# Multidimensional cube measurements

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

This report uses run `cube-20261001T041453929534Z-da641de6ef15` and executable source SHA-256 `69eb8feebdc26d4e72bd1160ed7e5cf695d5035a65f62edebc1a061061ba1fba`. The runner checked paired data, masks, initial parameters, polar-control curves and artifact hashes before publication. Independent [final acceptance](history/results/acceptance.json) replayed all 16 geometry cases, 80 erasure cases and six saved models successfully.

These measurements extend the original ChromaTensor pilot. They use distinct records and do not replace its findings.

The run measured 16 geometry cases and 80 erasure cases. Reconstruction models used 24 epochs on 1,024 training fields and 256 held-out fields with shape 3 × 3 × 3 × 3. The same seeds, masks, data, initial weights, architecture, and optimizer apply across each paired comparison.

| Representation | Seed | Initial held-out MSE | Final held-out MSE | Random-noise MSE | Training seconds |
| --- | --- | --- | --- | --- | --- |
| direct_rgb24 | 7 | 0.02779821 | 0.00329344 | 0.3249345 | 2.139 |
| holographic24 | 7 | 0.02778525 | 0.0119364 | 0.3145691 | 2.165 |
| numeric_polar24 | 7 | 0.02778525 | 0.0119364 | 0.3145691 | 1.902 |
| direct_rgb24 | 23 | 0.02630525 | 0.003099028 | 0.3275085 | 1.979 |
| holographic24 | 23 | 0.02629217 | 0.01083891 | 0.3132845 | 1.936 |
| numeric_polar24 | 23 | 0.02629217 | 0.01083891 | 0.3132845 | 2.007 |

Training seconds measure optimization batches only. Encoding, validation, checkpoint serialization, and final evaluation are excluded. Process peak memory includes optimization and per-epoch validation. Packed field payload is 243 bytes per 81-cell sample. Spectral encodings add eight bytes of magnitude-scale metadata. The dense neural input is 648 bytes per sample after numerical expansion and the mask. The model uses FP32 master weights and matrix operations with chroma8 fake quantization.

## Findings from this run

- The cube supports integer-only layer rotations in three through six dimensions, with exact four-turn and inverse restoration of cell codes and identities.
- The distributed field stores a phase8/magnitude16 full complex FFT. It uses floating point transforms and carries Hermitian redundancy. It is an implemented spectral representation, not a literal optical hologram.
- Every numeric polar control uses exactly the same 24-bit quantization as its chromatic field. Their inputs, training curves, and reconstruction outputs are identical. Color naming supplies no reconstruction advantage.
- At approximately 50% retained coordinates, mean held-out periodic-field MSE was 0.003196234 for direct RGB and 0.01138765 for the distributed field. These masks erase coordinates in different bases.
- Partial field erasure loses information. The trained reconstruction models evaluate held-out low-rank periodic fields from the same generator and a separate random-noise negative control. Their results do not establish recovery of arbitrary data from fragments.

Measured validation loss. Original artifact `artifacts/checkpoints/cube/cube-20261001T041453929534Z-da641de6ef15/cube-training.png` is omitted; regenerate it from source.

All erasure measurements and execution times are in the JSON record. Original artifact `results/runs/cube-20261001T041453929534Z-da641de6ef15.json` is omitted; regenerate it from source.. Geometry and CUDA correctness come from independent unit tests. Six-dimensional visual projection does not display six physical spatial axes.

The immutable generated report. Original artifact `artifacts/checkpoints/cube/cube-20261001T041453929534Z-da641de6ef15/CUBE_RESULTS.md` is omitted; regenerate it from source. preserves the original run output. Earlier source records remain historical.
