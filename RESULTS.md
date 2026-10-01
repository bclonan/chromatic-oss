# Measured results

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](docs/RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The current native hue release passed all six [acceptance gates](docs/history/results/acceptance.json) on source `0baada5d50c2fff69be4900a2b6820b714501b9573beada54059b0331bc7c6f9`. Its [million-value proof](docs/history/results/native-hue-verification.json) uses seed 7331 and three repeats. Seven operations match the independent integer reference exactly. A 32 by 32 contraction and actual RGB/D mutations pass. A separate isolated process reloads all one million values without Torch, NumPy or original input values.

The suite passed 776 Python tests with zero skips. Rust passed 26 tests on each of Windows and Linux. Eight clean installed-wheel environments passed on Python 3.11 through 3.14. The native payload occupies 4,000,000 bytes, equal to the uint32 control. Its saved file occupies 4,000,064 bytes. All measured native operations were slower than the NumPy integer control. [The native guide](docs/NATIVE_HUE.md) records scope, timing and limitations.

## Historical numerical controls

The results below describe the earlier scalar quantization and PyTorch research implementation. They are tied to their original source and do not measure native hue learning or certify the new native revision.

Run `bench-20261001T041728137087Z-970bb8116dbc` used NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu126, profile `pilot`.

The record contains 95 model experiments and 92 numerical representation/distribution comparisons.

Every value below comes from the saved experiment records. Accuracy is a fraction. Training time excludes validation and export; GPU timings synchronize the device. Storage is a diagnostic quantized export estimate unless a row explicitly names an actual checkpoint or integer inference model. Inference-only records have no training time.

| Task | Representation | Seed | Validation loss | Task metric | Training seconds | Inference samples/s |
| --- | --- | --- | --- | --- | --- | --- |
| scalar | fp32 | 7 | 0.000471547 | 0.000471547 | 1.539 | 2164538.9 |
| scalar | int8_integer_post_training | 7 | 0.000489336 | 0.000489336 | n/a | 422066.1 |
| scalar | fp16 | 7 | 0.000460867 | 0.000460867 | 1.822 | 363507.3 |
| scalar | bf16 | 7 | 0.000481276 | 0.000481276 | 1.569 | 943952.8 |
| scalar | int8 | 7 | 0.000532246 | 0.000532246 | 3.007 | 56196.3 |
| scalar | int4 | 7 | 0.00280104 | 0.00280104 | 2.654 | 59573.4 |
| scalar | fixed24 | 7 | 0.000471547 | 0.000471547 | 2.796 | 45438.2 |
| scalar | chroma8 | 7 | 0.000532246 | 0.000532246 | 3.653 | 35813.3 |
| scalar | rgb24 | 7 | 0.000471547 | 0.000471547 | 3.221 | 45490.1 |
| scalar | latent | 7 | 0.0984289 | 0.0984289 | 1.092 | 457036.8 |
| mnist | fp32 | 7 | 0.206578 | 0.94 | 0.689 | 2691054.3 |
| mnist | int8_integer_post_training | 7 | 0.206634 | 0.939 | n/a | 28924.7 |
| mnist | fp16 | 7 | 0.197888 | 0.94 | 0.999 | 690417.8 |
| mnist | bf16 | 7 | 0.204235 | 0.9365 | 0.886 | 612880.1 |
| mnist | int8 | 7 | 0.197203 | 0.9395 | 2.183 | 68052.7 |
| mnist | int4 | 7 | 0.194066 | 0.94 | 2.157 | 60065.0 |
| mnist | fixed24 | 7 | 0.207001 | 0.94 | 1.966 | 56901.8 |
| mnist | chroma8 | 7 | 0.197203 | 0.9395 | 2.682 | 39718.6 |
| mnist | rgb24 | 7 | 0.207001 | 0.94 | 2.578 | 37091.8 |
| mnist | chroma_weights | 7 | 0.20673 | 0.9375 | 1.704 | 75739.4 |
| mnist | chroma_activations | 7 | 0.208102 | 0.9375 | 1.926 | 82144.5 |
| cifar | fp32 | 7 | 1.6711 | 0.412 | 0.750 | 2275353.2 |
| cifar | int8_integer_post_training | 7 | 1.6712 | 0.411 | n/a | 6760.3 |
| cifar | fp16 | 7 | 1.67952 | 0.3995 | 0.886 | 865830.2 |
| cifar | bf16 | 7 | 1.66874 | 0.3995 | 0.865 | 686437.5 |
| cifar | int8 | 7 | 1.67226 | 0.4065 | 2.132 | 61469.9 |
| cifar | int4 | 7 | 1.69719 | 0.408 | 2.010 | 55249.2 |
| cifar | fixed24 | 7 | 1.6757 | 0.4065 | 2.188 | 60611.7 |
| cifar | chroma8 | 7 | 1.67226 | 0.4065 | 2.624 | 41234.7 |
| cifar | rgb24 | 7 | 1.6757 | 0.4065 | 3.064 | 25262.5 |
| cifar | chroma_weights | 7 | 1.66444 | 0.4195 | 1.916 | 81499.3 |
| cifar | chroma_activations | 7 | 1.6826 | 0.4095 | 2.074 | 64208.5 |
| language | fp32 | 7 | 2.35555 | 10.544 | 0.853 | 104460.8 |
| language | fp16 | 7 | 2.35556 | 10.544 | 1.321 | 57801.4 |
| language | bf16 | 7 | 2.35574 | 10.5459 | 1.283 | 41455.1 |
| language | int8 | 7 | 2.35564 | 10.5448 | 4.262 | 4248.7 |
| language | int4 | 7 | 2.3979 | 11.0001 | 4.124 | 4525.7 |
| language | fixed24 | 7 | 2.35555 | 10.544 | 4.049 | 4352.5 |
| language | chroma8 | 7 | 2.35564 | 10.5448 | 5.840 | 2526.9 |
| language | rgb24 | 7 | 2.35555 | 10.544 | 5.576 | 2836.7 |
| periodic | scalar8 | 7 | 0.000326533 | 0.00959374 | 0.885 | 1009264.7 |
| periodic | hue8 | 7 | 3.09989e-05 | 0.00638763 | 1.008 | 1000742.8 |
| periodic | numeric_circular8 | 7 | 3.09989e-05 | 0.00638763 | 1.178 | 750887.3 |
| periodic | raw_sincos64 | 7 | 4.39653e-06 | 0.00187088 | 1.017 | 1592039.9 |
| scalar | fp32 | 23 | 0.000555103 | 0.000555103 | 1.002 | 2707562.2 |
| scalar | int8_integer_post_training | 23 | 0.000589053 | 0.000589053 | n/a | 427807.5 |
| scalar | fp16 | 23 | 0.000549543 | 0.000549543 | 1.398 | 947902.4 |
| scalar | bf16 | 23 | 0.000551664 | 0.000551664 | 1.339 | 721431.6 |
| scalar | int8 | 23 | 0.000655306 | 0.000655306 | 2.682 | 65041.1 |
| scalar | int4 | 23 | 0.00293476 | 0.00293476 | 2.666 | 64756.0 |
| scalar | fixed24 | 23 | 0.000555104 | 0.000555104 | 2.582 | 63834.3 |
| scalar | chroma8 | 23 | 0.000655306 | 0.000655306 | 3.815 | 41482.4 |
| scalar | rgb24 | 23 | 0.000555104 | 0.000555104 | 3.735 | 35842.9 |
| scalar | latent | 23 | 0.110804 | 0.110804 | 1.365 | 379861.4 |
| mnist | fp32 | 23 | 0.214068 | 0.935 | 0.694 | 2589258.7 |
| mnist | int8_integer_post_training | 23 | 0.213977 | 0.935 | n/a | 28714.5 |
| mnist | fp16 | 23 | 0.213751 | 0.938 | 1.378 | 452001.3 |
| mnist | bf16 | 23 | 0.211951 | 0.9375 | 1.616 | 457936.1 |
| mnist | int8 | 23 | 0.214594 | 0.9355 | 2.122 | 62992.9 |
| mnist | int4 | 23 | 0.214786 | 0.9355 | 2.249 | 66653.0 |
| mnist | fixed24 | 23 | 0.213987 | 0.935 | 2.397 | 60458.3 |
| mnist | chroma8 | 23 | 0.214594 | 0.9355 | 3.062 | 34354.3 |
| mnist | rgb24 | 23 | 0.213987 | 0.935 | 3.107 | 36933.9 |
| mnist | chroma_weights | 23 | 0.215445 | 0.936 | 2.388 | 53456.7 |
| mnist | chroma_activations | 23 | 0.215729 | 0.937 | 3.142 | 47076.4 |
| cifar | fp32 | 23 | 1.66 | 0.4105 | 0.937 | 672569.2 |
| cifar | int8_integer_post_training | 23 | 1.65953 | 0.4095 | n/a | 6143.9 |
| cifar | fp16 | 23 | 1.66393 | 0.406 | 1.050 | 798129.4 |
| cifar | bf16 | 23 | 1.66364 | 0.4075 | 1.026 | 437921.2 |
| cifar | int8 | 23 | 1.63136 | 0.4215 | 2.120 | 43411.2 |
| cifar | int4 | 23 | 1.68284 | 0.396 | 2.060 | 59974.2 |
| cifar | fixed24 | 23 | 1.66114 | 0.4095 | 2.115 | 60398.9 |
| cifar | chroma8 | 23 | 1.63136 | 0.4215 | 2.842 | 35816.9 |
| cifar | rgb24 | 23 | 1.66114 | 0.4095 | 2.613 | 39240.6 |
| cifar | chroma_weights | 23 | 1.65987 | 0.4025 | 1.649 | 68966.8 |
| cifar | chroma_activations | 23 | 1.66782 | 0.3945 | 1.819 | 70022.6 |
| language | fp32 | 23 | 2.37401 | 10.7404 | 0.936 | 97744.2 |
| language | fp16 | 23 | 2.37401 | 10.7404 | 1.326 | 45486.9 |
| language | bf16 | 23 | 2.37405 | 10.7408 | 1.326 | 51959.9 |
| language | int8 | 23 | 2.37421 | 10.7425 | 4.238 | 3854.2 |
| language | int4 | 23 | 2.41963 | 11.2417 | 4.026 | 4119.8 |
| language | fixed24 | 23 | 2.37401 | 10.7404 | 4.059 | 4515.3 |
| language | chroma8 | 23 | 2.37421 | 10.7425 | 5.966 | 3053.5 |
| language | rgb24 | 23 | 2.37401 | 10.7404 | 5.588 | 2473.4 |
| periodic | scalar8 | 23 | 0.000569954 | 0.013288 | 0.951 | 983782.9 |
| periodic | hue8 | 23 | 3.18563e-05 | 0.00644183 | 1.185 | 2620266.2 |
| periodic | numeric_circular8 | 23 | 3.18563e-05 | 0.00644183 | 1.369 | 2072706.7 |
| periodic | raw_sincos64 | 23 | 5.6032e-06 | 0.00216412 | 2.378 | 1045666.2 |
| attention | fp32_attention_dot | 7 | 0.00454871 | 0.00454871 | 1.404 | 31367.9 |
| attention | chroma8_attention_dot | 7 | 0.00450677 | 0.00450677 | 3.923 | 6733.2 |
| attention | chroma8_attention_codebook | 7 | 0.00522737 | 0.00522737 | 4.084 | 5698.0 |
| attention | fp32_attention_dot | 23 | 0.00337704 | 0.00337704 | 1.188 | 63689.2 |
| attention | chroma8_attention_dot | 23 | 0.00337809 | 0.00337809 | 3.549 | 7637.1 |
| attention | chroma8_attention_codebook | 23 | 0.00405204 | 0.00405204 | 4.336 | 3690.7 |
| decoder | learned_rgb | 7 | 0.000530141 | 0.000530141 | 1.496 | 3217899.6 |

## Paired controls

| Task | Comparison | Mean metric difference | Seeds |
| --- | --- | --- | --- |
| cifar | chroma8 minus int8 | 0 | 2 |
| cifar | rgb24 minus fixed24 | 0 | 2 |
| cifar | chroma8 minus fp32 | 0.00275 | 2 |
| language | chroma8 minus int8 | 0 | 2 |
| language | rgb24 minus fixed24 | 0 | 2 |
| language | chroma8 minus fp32 | 0.00152454795 | 2 |
| mnist | chroma8 minus int8 | 0 | 2 |
| mnist | rgb24 minus fixed24 | 0 | 2 |
| mnist | chroma8 minus fp32 | 0 | 2 |
| periodic | hue8 minus numeric_circular8 | 0 | 2 |
| scalar | chroma8 minus int8 | 0 | 2 |
| scalar | rgb24 minus fixed24 | 0 | 2 |
| scalar | chroma8 minus fp32 | 8.04505507e-05 | 2 |

## Precision on 1,000,000 bounded values per distribution

The suite records unclipped out-of-domain sample counts, but evaluates quantization on the same clipped values for each bounded codec. Hue uses circular error, so its MAE does not compare directly with linear error.

| Distribution | Codec | MAE | Max error | Actual code bits/value | Metadata bytes |
| --- | --- | --- | --- | --- | --- |
| uniform | rgb24 | 2.979452e-08 | 5.960462e-08 | 24.0 | 70 |
| uniform | rgbd32 | 9.919382e-09 | 2.980232e-08 | 32.0 | 72 |
| uniform | hsv24_effective8 | 0.001962397 | 0.003921568 | 24.0 | 48 |
| uniform | hue8 | 0.001951511 | 0.00390625 | 8.0 | 59 |
| uniform | int4 | 0.03333971 | 0.06666658 | 8.0 | 85 |
| uniform | int8 | 0.001962397 | 0.003921568 | 8.0 | 85 |
| uniform | fixed24 | 2.979452e-08 | 5.960462e-08 | 32.0 | 86 |
| uniform | residual16_depth3 | 0.0001480728 | 0.0002962959 | 32.0 | 112 |
| uniform | adaptive_residual | 0.0005555336 | 0.001999999 | 32.0 | 113 |
| uniform | learned_rgbd | 0.009853357 | 0.07987454 | 32.0 | 3576 |
| uniform | learned_vocab16 | 0.08642131 | 0.3739482 | 8.0 | 896 |
| uniform | learned_vocab64 | 0.04148552 | 0.2503562 | 8.0 | 2640 |
| uniform | learned_vocab256 | 0.01826523 | 0.1506525 | 8.0 | 9684 |
| uniform | learned_vocab1024 | 0.008327332 | 0.0753667 | 16.0 | 38017 |
| uniform | learned_vocab4096 | 0.003731494 | 0.0448375 | 16.0 | 151665 |
| uniform | learned_vocab65536 | 0.0005018986 | 0.01015384 | 16.0 | 2417980 |
| uniform | uniform_vocab64 | 0.007936551 | 0.01587299 | 8.0 | 85 |
| uniform | uniform_vocab1024 | 0.0004886182 | 0.0009775162 | 16.0 | 86 |
| uniform | uniform_vocab4096 | 0.0001222207 | 0.0002442002 | 16.0 | 86 |
| uniform | uniform_vocab65536 | 7.638211e-06 | 1.525902e-05 | 16.0 | 86 |
| uniform | float32 | 9.919382e-09 | 2.980232e-08 | 32.0 | 0 |
| uniform | float16 | 8.135102e-05 | 0.0002441684 | 16.0 | 0 |
| uniform | bfloat16 | 0.000651021 | 0.001953149 | 16.0 | 0 |
| normal | rgb24 | 2.979744e-08 | 5.960461e-08 | 24.0 | 70 |
| normal | rgbd32 | 3.429693e-09 | 2.979883e-08 | 32.0 | 72 |
| normal | hsv24_effective8 | 0.00196126 | 0.00392156 | 24.0 | 48 |
| normal | hue8 | 0.001952881 | 0.003906245 | 8.0 | 59 |
| normal | int4 | 0.03334709 | 0.06666666 | 8.0 | 85 |
| normal | int8 | 0.00196126 | 0.00392156 | 8.0 | 85 |
| normal | fixed24 | 2.979744e-08 | 5.960461e-08 | 32.0 | 86 |
| normal | residual16_depth3 | 0.0001480748 | 0.0002962962 | 32.0 | 112 |
| normal | adaptive_residual | 0.000554675 | 0.001999997 | 32.0 | 113 |
| normal | learned_rgbd | 0.0006568311 | 0.07282221 | 32.0 | 3576 |
| normal | learned_vocab16 | 0.0163516 | 0.373951 | 8.0 | 896 |
| normal | learned_vocab64 | 0.004030285 | 0.2503591 | 8.0 | 2640 |
| normal | learned_vocab256 | 0.001014864 | 0.1300392 | 8.0 | 9684 |
| normal | learned_vocab1024 | 0.0002546436 | 0.0633474 | 16.0 | 38017 |
| normal | learned_vocab4096 | 6.482559e-05 | 0.04060061 | 16.0 | 151665 |
| normal | learned_vocab65536 | 5.522469e-06 | 0.005089338 | 16.0 | 2417980 |
| normal | uniform_vocab64 | 0.007939722 | 0.01587298 | 8.0 | 85 |
| normal | uniform_vocab1024 | 0.000488765 | 0.0009775168 | 16.0 | 86 |
| normal | uniform_vocab4096 | 0.0001221899 | 0.0002441998 | 16.0 | 86 |
| normal | uniform_vocab65536 | 7.623716e-06 | 1.525902e-05 | 16.0 | 86 |
| normal | float32 | 3.429693e-09 | 2.979883e-08 | 32.0 | 0 |
| normal | float16 | 2.809913e-05 | 0.0002441379 | 16.0 | 0 |
| normal | bfloat16 | 0.0002247853 | 0.001953085 | 16.0 | 0 |
| laplace | rgb24 | 2.977243e-08 | 5.960459e-08 | 24.0 | 70 |
| laplace | rgbd32 | 0 | 0 | 32.0 | 72 |
| laplace | hsv24_effective8 | 0.00195816 | 0.003921567 | 24.0 | 48 |
| laplace | hue8 | 0.00195106 | 0.003906239 | 8.0 | 59 |
| laplace | int4 | 0.03383648 | 0.06666666 | 8.0 | 85 |
| laplace | int8 | 0.00195816 | 0.003921567 | 8.0 | 85 |
| laplace | fixed24 | 2.977243e-08 | 5.960459e-08 | 32.0 | 86 |
| laplace | residual16_depth3 | 0.000147897 | 0.0002962959 | 32.0 | 112 |
| laplace | adaptive_residual | 0.0005543159 | 0.001999996 | 32.0 | 113 |
| laplace | learned_rgbd | 0.0009674052 | 0.07987463 | 32.0 | 3576 |
| laplace | learned_vocab16 | 0.01766431 | 0.373951 | 8.0 | 896 |
| laplace | learned_vocab64 | 0.005107542 | 0.2503591 | 8.0 | 2640 |
| laplace | learned_vocab256 | 0.001615929 | 0.1506526 | 8.0 | 9684 |
| laplace | learned_vocab1024 | 0.0005341713 | 0.0753668 | 16.0 | 38017 |
| laplace | learned_vocab4096 | 0.0001709254 | 0.0446251 | 16.0 | 151665 |
| laplace | learned_vocab65536 | 1.418821e-05 | 0.01004165 | 16.0 | 2417980 |
| laplace | uniform_vocab64 | 0.007933973 | 0.01587301 | 8.0 | 85 |
| laplace | uniform_vocab1024 | 0.0004883583 | 0.0009775154 | 16.0 | 86 |
| laplace | uniform_vocab4096 | 0.0001220165 | 0.0002441999 | 16.0 | 86 |
| laplace | uniform_vocab65536 | 7.623248e-06 | 1.525902e-05 | 16.0 | 86 |
| laplace | float32 | 0 | 0 | 32.0 | 0 |
| laplace | float16 | 2.608001e-05 | 0.0002441406 | 16.0 | 0 |
| laplace | bfloat16 | 0.0002088638 | 0.001953065 | 16.0 | 0 |
| bimodal | rgb24 | 2.980925e-08 | 5.960461e-08 | 24.0 | 70 |
| bimodal | rgbd32 | 1.411984e-08 | 2.980225e-08 | 32.0 | 72 |
| bimodal | hsv24_effective8 | 0.00195953 | 0.003921567 | 24.0 | 48 |
| bimodal | hue8 | 0.001950843 | 0.003906249 | 8.0 | 59 |
| bimodal | int4 | 0.03330784 | 0.06666665 | 8.0 | 85 |
| bimodal | int8 | 0.00195953 | 0.003921567 | 8.0 | 85 |
| bimodal | fixed24 | 2.980925e-08 | 5.960461e-08 | 32.0 | 86 |
| bimodal | residual16_depth3 | 0.000148112 | 0.0002962959 | 32.0 | 112 |
| bimodal | adaptive_residual | 0.0005555947 | 0.002 | 32.0 | 113 |
| bimodal | learned_rgbd | 0.007013988 | 0.0687898 | 32.0 | 3576 |
| bimodal | learned_vocab16 | 0.04893374 | 0.3564807 | 8.0 | 896 |
| bimodal | learned_vocab64 | 0.0293492 | 0.2253779 | 8.0 | 2640 |
| bimodal | learned_vocab256 | 0.01312846 | 0.13346 | 8.0 | 9684 |
| bimodal | learned_vocab1024 | 0.004017603 | 0.06401862 | 16.0 | 38017 |
| bimodal | learned_vocab4096 | 0.001016666 | 0.04378984 | 16.0 | 151665 |
| bimodal | learned_vocab65536 | 8.732791e-05 | 0.009298676 | 16.0 | 2417980 |
| bimodal | uniform_vocab64 | 0.007926917 | 0.015873 | 8.0 | 85 |
| bimodal | uniform_vocab1024 | 0.0004888534 | 0.0009775147 | 16.0 | 86 |
| bimodal | uniform_vocab4096 | 0.0001221824 | 0.0002441999 | 16.0 | 86 |
| bimodal | uniform_vocab65536 | 7.634514e-06 | 1.525902e-05 | 16.0 | 86 |
| bimodal | float32 | 1.411984e-08 | 2.980225e-08 | 32.0 | 0 |
| bimodal | float16 | 0.0001156442 | 0.0002441701 | 16.0 | 0 |
| bimodal | bfloat16 | 0.0009242042 | 0.001953146 | 16.0 | 0 |

## Operator measurements

| Method | MAE | Batch latency ms | Values/s |
| --- | --- | --- | --- |
| decode_compute_encode | 2.932211e-09 | 0.30338 | 27002439.8 |
| lookup | 2.932211e-09 | 0.20627 | 39714936.9 |
| direct_integer_grid | 2.932211e-09 | 0.07579 | 108088145.4 |
| learned | 2.932211e-09 | 1.58233 | 5177175.4 |

## Actual file storage

| Format | File bytes | Separate metadata bytes | Bits/value including metadata | Reconstruction MAE |
| --- | --- | --- | --- | --- |
| rgb24_raw.ctensor | 49433 | 0 | 24.137 | 2.972519e-08 |
| rgb24_zlib.ctensor | 48914 | 0 | 23.884 | 2.972519e-08 |
| rgbd32_raw.ctensor | 65823 | 0 | 32.14 | 0 |
| rgbd32_zlib.ctensor | 60882 | 0 | 29.728 | 0 |
| int8_raw.ctensor | 16678 | 0 | 8.1436 | 0.001962124 |
| int8_zlib.ctensor | 14154 | 0 | 6.9111 | 0.001962124 |
| int4_raw.ctensor | 16678 | 0 | 8.1436 | 0.03336917 |
| int4_zlib.ctensor | 7053 | 0 | 3.4438 | 0.03336917 |
| residual_raw.ctensor | 65856 | 0 | 32.156 | 0.0005608354 |
| residual_zlib.ctensor | 28043 | 0 | 13.693 | 0.0005608354 |
| float32.npy | 65664 | 0 | 32.062 | 0 |
| float32.npz | 60856 | 0 | 29.715 | 0 |
| float16.npy | 32896 | 0 | 16.062 | 2.813601e-05 |
| float16.npz | 30412 | 0 | 14.85 | 2.813601e-05 |
| fp32.pt | 67028 | 0 | 32.729 | 0 |
| packed_int4.bin | 8192 | 95 | 4.0464 | 0.03336916 |
| packed_fixed24.bin | 49152 | 95 | 24.046 | 2.96989e-08 |

## Instrumentation

The JSON includes per-model matrix operation estimates, live master bytes, runtime RGB buffers, diagnostic exports, and partial PyTorch CPU profiler observations from trained scalar checkpoints. The estimates exclude uncounted operations and are not hardware instruction counts. See docs/MEASUREMENT.md for the complete scope.

## Interpretation

- cifar: chroma8 minus int8 mean paired accuracy difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- cifar: rgb24 minus fixed24 mean paired accuracy difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- language: chroma8 minus int8 mean paired perplexity difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- language: rgb24 minus fixed24 mean paired perplexity difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- mnist: chroma8 minus int8 mean paired accuracy difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- mnist: rgb24 minus fixed24 mean paired accuracy difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- periodic: hue8 minus numeric_circular8 mean paired angular_mae_radians difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- scalar: chroma8 minus int8 mean paired validation_loss difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- scalar: rgb24 minus fixed24 mean paired validation_loss difference 0 across 2 seeds. These controls share quantization levels, so color labels alone do not establish an advantage.
- RGB24 is a fixed-point quantizer with color byte labels. RGBD32 preserves finite FP32 values by exposing sign, fraction, and exponent bytes. Neither removes binary arithmetic.
- On clipped normal values, the separately fitted 256-centroid learned palette has MAE 0.00101486, versus 0.00196126 for uniform INT8. This is distribution-aware scalar vector quantization; learned tables have additional storage cost.
- The jointly trained neural RGB codec reduced held-out MSE from 0.211442 to 0.000530141. Its held-out MAE 0.020719 exceeds uniform8 MAE 0.00195123 and uniform24 MAE 2.99607e-08 on the identical values. Learned color coordinates worked, but these measurements do not justify their extra model/storage cost.
- The causal phase attention ablation produced mean held-out MSE chroma8_attention_codebook=0.0046397, chroma8_attention_dot=0.00394243, fp32_attention_dot=0.00396287. Codebook inference still uses floating point nearest-code assignment, similarity-table construction, softmax, and value aggregation.
- Flipping a palette payload bit produced an invalid chroma8 symbol for 100.0% of the tested values. The palette supplies no inherent bit-corruption robustness; checksums and vocabulary validation remain necessary.
- Quantization-aware networks use floating point master parameters, optimizer state, matrix products, normalization, and softmax. The operator lookup and grid-add paths perform integer operations after their tables or scale metadata have been prepared.
- The executed profile is a seeded research pilot. Limited training, small models, and two seeds do not establish state-of-the-art accuracy, novel mathematics, end-to-end integer training, or a general speed advantage.
