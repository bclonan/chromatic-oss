Historical result from the original research checkout. See docs/RELEASE_AUDIT.md for current verification.

# Measured results

Run `20261001T001139Z` used NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu126, profile `smoke`.

The record contains 17 model experiments and 92 numerical representation/distribution comparisons.

Every value below comes from the saved experiment records. Accuracy is a fraction. Training time excludes validation and export; GPU timings synchronize the device. Storage is a diagnostic quantized export estimate unless a row explicitly names an actual checkpoint or integer inference model. Inference-only records have no training time.

| Task | Representation | Seed | Validation loss | Task metric | Training seconds | Inference samples/s |
| --- | --- | --- | --- | --- | --- | --- |
| scalar | fp32 | 7 | 0.218647 | 0.218647 | 0.077 | 1536061.4 |
| scalar | int8_integer_post_training | 7 | 0.21867 | 0.21867 | n/a | 175124.2 |
| scalar | fp16 | 7 | 0.218655 | 0.218655 | 0.032 | 166443.4 |
| scalar | bf16 | 7 | 0.218687 | 0.218687 | 0.029 | 175651.7 |
| scalar | int8 | 7 | 0.21871 | 0.21871 | 0.022 | 191712.9 |
| scalar | int4 | 7 | 0.224179 | 0.224179 | 0.017 | 224612.6 |
| scalar | fixed24 | 7 | 0.218647 | 0.218647 | 0.011 | 208353.7 |
| scalar | chroma8 | 7 | 0.21871 | 0.21871 | 0.021 | 67961.8 |
| scalar | rgb24 | 7 | 0.218647 | 0.218647 | 0.029 | 54847.5 |
| periodic | scalar8 | 7 | 0.391014 | 0.754155 | 0.009 | 1557461.8 |
| periodic | hue8 | 7 | 0.418188 | 0.967905 | 0.009 | 1347013.9 |
| periodic | numeric_circular8 | 7 | 0.418188 | 0.967905 | 0.010 | 1449357.4 |
| periodic | raw_sincos64 | 7 | 0.418192 | 0.967051 | 0.007 | 1560499.8 |
| attention | fp32_attention_dot | 7 | 0.00430062 | 0.00430062 | 1.278 | 45115.5 |
| attention | chroma8_attention_dot | 7 | 0.00433348 | 0.00433348 | 2.745 | 8611.0 |
| attention | chroma8_attention_codebook | 7 | 0.00507297 | 0.00507297 | 4.091 | 9213.4 |
| decoder | learned_rgb | 7 | 0.00340602 | 0.00340602 | 0.154 | 1188045.3 |

## Paired controls

| Task | Comparison | Mean metric difference | Seeds |
| --- | --- | --- | --- |
| periodic | hue8 minus numeric_circular8 | 0 | 1 |
| scalar | chroma8 minus int8 | 0 | 1 |
| scalar | rgb24 minus fixed24 | 0 | 1 |
| scalar | chroma8 minus fp32 | 6.26146793e-05 | 1 |

## Precision on 4,096 bounded values per distribution

The suite records unclipped out-of-domain sample counts, but evaluates quantization on the same clipped values for each bounded codec. Hue uses circular error, so its MAE does not compare directly with linear error.

| Distribution | Codec | MAE | Max error | Actual code bits/value | Metadata bytes |
| --- | --- | --- | --- | --- | --- |
| uniform | rgb24 | 2.980561e-08 | 5.960283e-08 | 24.0 | 70 |
| uniform | rgbd32 | 9.810442e-09 | 2.978075e-08 | 32.0 | 72 |
| uniform | hsv24_effective8 | 0.001982936 | 0.003920636 | 24.0 | 48 |
| uniform | hue8 | 0.001942703 | 0.003903352 | 8.0 | 59 |
| uniform | int4 | 0.03365485 | 0.06665248 | 8.0 | 85 |
| uniform | int8 | 0.001982936 | 0.003920636 | 8.0 | 85 |
| uniform | fixed24 | 2.980561e-08 | 5.960283e-08 | 32.0 | 86 |
| uniform | residual16_depth3 | 0.0001491783 | 0.0002960941 | 32.0 | 112 |
| uniform | adaptive_residual | 0.0005620067 | 0.001999384 | 32.0 | 113 |
| uniform | learned_rgbd | 0.01021011 | 0.1017722 | 32.0 | 3522 |
| uniform | learned_vocab16 | 0.08631036 | 0.3754245 | 8.0 | 870 |
| uniform | learned_vocab64 | 0.04110732 | 0.2525747 | 8.0 | 2613 |
| uniform | learned_vocab256 | 0.02019831 | 0.1596305 | 8.0 | 9660 |
| uniform | learned_vocab1024 | 0.008788717 | 0.09083537 | 16.0 | 38026 |
| uniform | learned_vocab4096 | 0.003645419 | 0.04987177 | 16.0 | 151683 |
| uniform | learned_vocab65536 | 0.0006350221 | 0.01789164 | 16.0 | 2418075 |
| uniform | uniform_vocab64 | 0.007911832 | 0.01587082 | 8.0 | 85 |
| uniform | uniform_vocab1024 | 0.0004842932 | 0.0009771274 | 16.0 | 86 |
| uniform | uniform_vocab4096 | 0.0001216339 | 0.0002441812 | 16.0 | 86 |
| uniform | uniform_vocab65536 | 7.510305e-06 | 1.525455e-05 | 16.0 | 86 |
| uniform | float32 | 9.810442e-09 | 2.978075e-08 | 32.0 | 0 |
| uniform | float16 | 7.946256e-05 | 0.000244158 | 16.0 | 0 |
| uniform | bfloat16 | 0.0006464673 | 0.001952454 | 16.0 | 0 |
| normal | rgb24 | 2.956779e-08 | 5.958881e-08 | 24.0 | 70 |
| normal | rgbd32 | 3.492093e-09 | 2.94189e-08 | 32.0 | 72 |
| normal | hsv24_effective8 | 0.001934401 | 0.003921358 | 24.0 | 48 |
| normal | hue8 | 0.001968689 | 0.003905981 | 8.0 | 59 |
| normal | int4 | 0.03330971 | 0.06666016 | 8.0 | 85 |
| normal | int8 | 0.001934401 | 0.003921358 | 8.0 | 85 |
| normal | fixed24 | 2.956779e-08 | 5.958881e-08 | 32.0 | 86 |
| normal | residual16_depth3 | 0.0001507487 | 0.0002961571 | 32.0 | 112 |
| normal | adaptive_residual | 0.0005508973 | 0.001999593 | 32.0 | 113 |
| normal | learned_rgbd | 0.000648839 | 0.03258772 | 32.0 | 3522 |
| normal | learned_vocab16 | 0.01617173 | 0.08899643 | 8.0 | 870 |
| normal | learned_vocab64 | 0.004054018 | 0.07386917 | 8.0 | 2613 |
| normal | learned_vocab256 | 0.0009890716 | 0.03853548 | 8.0 | 9660 |
| normal | learned_vocab1024 | 0.0002496845 | 0.01484943 | 16.0 | 38026 |
| normal | learned_vocab4096 | 6.475083e-05 | 0.003511856 | 16.0 | 151683 |
| normal | learned_vocab65536 | 5.383346e-06 | 0.0002349715 | 16.0 | 2418075 |
| normal | uniform_vocab64 | 0.007904307 | 0.0158627 | 8.0 | 85 |
| normal | uniform_vocab1024 | 0.0004879729 | 0.0009773987 | 16.0 | 86 |
| normal | uniform_vocab4096 | 0.0001241474 | 0.0002441973 | 16.0 | 86 |
| normal | uniform_vocab65536 | 7.619129e-06 | 1.52576e-05 | 16.0 | 86 |
| normal | float32 | 3.492093e-09 | 2.94189e-08 | 32.0 | 0 |
| normal | float16 | 2.756837e-05 | 0.0002368504 | 16.0 | 0 |
| normal | bfloat16 | 0.0002198738 | 0.001906638 | 16.0 | 0 |
| laplace | rgb24 | 3.004137e-08 | 5.95728e-08 | 24.0 | 70 |
| laplace | rgbd32 | 0 | 0 | 32.0 | 72 |
| laplace | hsv24_effective8 | 0.001953916 | 0.003920605 | 24.0 | 48 |
| laplace | hue8 | 0.001965379 | 0.003905565 | 8.0 | 59 |
| laplace | int4 | 0.03437292 | 0.06666144 | 8.0 | 85 |
| laplace | int8 | 0.001953916 | 0.003920605 | 8.0 | 85 |
| laplace | fixed24 | 3.004137e-08 | 5.95728e-08 | 32.0 | 86 |
| laplace | residual16_depth3 | 0.0001482518 | 0.0002958276 | 32.0 | 112 |
| laplace | adaptive_residual | 0.0005564598 | 0.001997415 | 32.0 | 113 |
| laplace | learned_rgbd | 0.001021728 | 0.1021716 | 32.0 | 3522 |
| laplace | learned_vocab16 | 0.01821701 | 0.3758239 | 8.0 | 870 |
| laplace | learned_vocab64 | 0.005191043 | 0.2529742 | 8.0 | 2613 |
| laplace | learned_vocab256 | 0.001663197 | 0.1600299 | 8.0 | 9660 |
| laplace | learned_vocab1024 | 0.0005733244 | 0.0912348 | 16.0 | 38026 |
| laplace | learned_vocab4096 | 0.0001616067 | 0.04957289 | 16.0 | 151683 |
| laplace | learned_vocab65536 | 1.814671e-05 | 0.00935173 | 16.0 | 2418075 |
| laplace | uniform_vocab64 | 0.007958764 | 0.01586948 | 8.0 | 85 |
| laplace | uniform_vocab1024 | 0.0004833347 | 0.0009774966 | 16.0 | 86 |
| laplace | uniform_vocab4096 | 0.0001206106 | 0.0002441447 | 16.0 | 86 |
| laplace | uniform_vocab65536 | 7.55419e-06 | 1.525075e-05 | 16.0 | 86 |
| laplace | float32 | 0 | 0 | 32.0 | 0 |
| laplace | float16 | 2.725703e-05 | 0.0002434254 | 16.0 | 0 |
| laplace | bfloat16 | 0.00021769 | 0.001951218 | 16.0 | 0 |
| bimodal | rgb24 | 2.978584e-08 | 5.959461e-08 | 24.0 | 70 |
| bimodal | rgbd32 | 1.425201e-08 | 2.980229e-08 | 32.0 | 72 |
| bimodal | hsv24_effective8 | 0.001960322 | 0.003921079 | 24.0 | 48 |
| bimodal | hue8 | 0.001940541 | 0.003906211 | 8.0 | 59 |
| bimodal | int4 | 0.03400599 | 0.06666554 | 8.0 | 85 |
| bimodal | int8 | 0.001960322 | 0.003921079 | 8.0 | 85 |
| bimodal | fixed24 | 2.978584e-08 | 5.959461e-08 | 32.0 | 86 |
| bimodal | residual16_depth3 | 0.0001482655 | 0.0002960903 | 32.0 | 112 |
| bimodal | adaptive_residual | 0.0005586773 | 0.001999725 | 32.0 | 113 |
| bimodal | learned_rgbd | 0.007069523 | 0.05317762 | 32.0 | 3522 |
| bimodal | learned_vocab16 | 0.04889236 | 0.2325682 | 8.0 | 870 |
| bimodal | learned_vocab64 | 0.02973371 | 0.09456113 | 8.0 | 2613 |
| bimodal | learned_vocab256 | 0.01298826 | 0.06388442 | 8.0 | 9660 |
| bimodal | learned_vocab1024 | 0.004026426 | 0.04330012 | 16.0 | 38026 |
| bimodal | learned_vocab4096 | 0.001055267 | 0.01858636 | 16.0 | 151683 |
| bimodal | learned_vocab65536 | 8.939566e-05 | 0.001680695 | 16.0 | 2418075 |
| bimodal | uniform_vocab64 | 0.007943057 | 0.01586757 | 8.0 | 85 |
| bimodal | uniform_vocab1024 | 0.0004849184 | 0.0009774669 | 16.0 | 86 |
| bimodal | uniform_vocab4096 | 0.0001217894 | 0.0002441831 | 16.0 | 86 |
| bimodal | uniform_vocab65536 | 7.656383e-06 | 1.525798e-05 | 16.0 | 86 |
| bimodal | float32 | 1.425201e-08 | 2.980229e-08 | 32.0 | 0 |
| bimodal | float16 | 0.0001170201 | 0.0002440173 | 16.0 | 0 |
| bimodal | bfloat16 | 0.0009246546 | 0.001952971 | 16.0 | 0 |

## Operator measurements

| Method | MAE | Batch latency ms | Values/s |
| --- | --- | --- | --- |
| decode_compute_encode | 2.932211e-09 | 0.43824 | 18692953.8 |
| lookup | 2.932211e-09 | 0.23311 | 35142206.7 |
| direct_integer_grid | 2.932211e-09 | 0.13792 | 59396750.7 |
| learned | 2.932211e-09 | 2.27745 | 3597005.4 |

## Actual file storage

| Format | File bytes | Separate metadata bytes | Bits/value including metadata | Reconstruction MAE |
| --- | --- | --- | --- | --- |
| rgb24_raw.ctensor | 12567 | 0 | 24.545 | 2.974469e-08 |
| rgb24_zlib.ctensor | 12443 | 0 | 24.303 | 2.974469e-08 |
| rgbd32_raw.ctensor | 16669 | 0 | 32.557 | 0 |
| rgbd32_zlib.ctensor | 15433 | 0 | 30.143 | 0 |
| int8_raw.ctensor | 4386 | 0 | 8.5664 | 0.001934401 |
| int8_zlib.ctensor | 3790 | 0 | 7.4023 | 0.001934401 |
| int4_raw.ctensor | 4386 | 0 | 8.5664 | 0.03330971 |
| int4_zlib.ctensor | 1982 | 0 | 3.8711 | 0.03330971 |
| residual_raw.ctensor | 16702 | 0 | 32.621 | 0.0005508974 |
| residual_zlib.ctensor | 7489 | 0 | 14.627 | 0.0005508974 |
| float32.npy | 16512 | 0 | 32.25 | 0 |
| float32.npz | 15397 | 0 | 30.072 | 0 |
| float16.npy | 8320 | 0 | 16.25 | 2.756833e-05 |
| float16.npz | 7791 | 0 | 15.217 | 2.756833e-05 |
| fp32.pt | 17876 | 0 | 34.914 | 0 |
| packed_int4.bin | 2048 | 94 | 4.1836 | 0.03330971 |
| packed_fixed24.bin | 12288 | 94 | 24.184 | 2.965783e-08 |

## Instrumentation

The JSON includes per-model matrix operation estimates, live master bytes, runtime RGB buffers, diagnostic exports, and partial PyTorch CPU profiler observations from trained scalar checkpoints. The estimates exclude uncounted operations and are not hardware instruction counts. See docs/MEASUREMENT.md for the complete scope.

## Interpretation

- periodic: hue8 minus numeric_circular8 mean paired angular_mae_radians difference 0 across 1 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- scalar: chroma8 minus int8 mean paired validation_loss difference 0 across 1 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- scalar: rgb24 minus fixed24 mean paired validation_loss difference 0 across 1 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- RGB24 is a fixed-point quantizer with color byte labels. RGBD32 preserves finite FP32 values by exposing sign, fraction, and exponent bytes. Neither removes binary arithmetic.
- On clipped normal values, the separately fitted 256-centroid learned palette has MAE 0.000989072, versus 0.0019344 for uniform INT8. This is distribution-aware scalar vector quantization; learned tables have additional storage cost.
- The jointly trained neural RGB codec reduced held-out MSE from 0.207178 to 0.00340602. Its held-out MAE 0.0378801 exceeds uniform8 MAE 0.00194784 and uniform24 MAE 2.94297e-08 on the identical values. Learned color coordinates worked, but these measurements do not justify their extra model/storage cost.
- The causal phase attention ablation produced mean held-out MSE chroma8_attention_codebook=0.00507297, chroma8_attention_dot=0.00433348, fp32_attention_dot=0.00430062. Codebook inference still uses floating point nearest-code assignment, similarity-table construction, softmax, and value aggregation.
- Flipping a palette payload bit produced an invalid chroma8 symbol for 100.0% of the tested values. The palette supplies no inherent bit-corruption robustness; checksums and vocabulary validation remain necessary.
- Quantization-aware networks use floating point master parameters, optimizer state, matrix products, normalization, and softmax. The operator lookup and grid-add paths perform integer operations after their tables or scale metadata have been prepared.
- The executed profile is a seeded research pilot. Limited training, small models, and two seeds do not establish state-of-the-art accuracy, novel mathematics, end-to-end integer training, or a general speed advantage.
