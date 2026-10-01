# Experiment design

Historical research document. Recorded metrics and source identities describe the original checkout. Use [the release audit](docs/RELEASE_AUDIT.md) and the current setup/testing guides for this sanitized candidate. Some original companion artifacts are omitted; regenerate them with the included scripts.

The hypothesis is that discrete color representations can support useful learning and that circular geometry, residual precision, or learned color structure may offer measurable benefits over equally sized numerical codes. A color label by itself is not a different numerical method.

The pilot uses seeds 7 and 23. Each paired representation gets the same data hash, initial master weights, architecture, optimizer settings, batch permutation generator, and training budget. FP16/BF16 use autocast with FP32 master parameters. INT8/INT4 train with fake quantization. Separate CPU post-training INT8 records perform actual INT32 linear products and FP32 scaling.

| Experiment | Data and model | Main task metric | Controls |
| --- | --- | --- | --- |
| Scalar | 4,096 train, 1,024 held out, two-hidden-layer 64-unit MLP, three analytic targets | Held-out MSE | FP32, FP16, BF16, INT8, INT4, fixed24 |
| MNIST | 12,000 stratified official train, 2,000 official test, 128-unit MLP | Classification accuracy | Same controls |
| CIFAR-10 | Same pilot sample counts, flattened pixels, 128-unit MLP | Classification accuracy | Same controls |
| Language | 2,048 train and 512 validation windows, 32 characters, 48-unit causal Transformer | Cross entropy and perplexity | Same controls |
| Periodic | Phase rotation with nonlinear phase update, MLP predicts sin/cos | Mean circular angle error | Linear 8-bit phase, equal-level numeric circular codes, original sin/cos |
| Precision | 1,000,000 values each, uniform, normal, Laplace, bimodal | MAE, MSE, maximum error, relative error | FP formats, INT4/8, fixed24, uniform vocabularies |
| Attention | Causal analytic phase sequences, conventional and codebook attention | MSE and angular error | Shared projection initialization and data |

The scalar targets are `sin(pi*x0)`, `x0^2`, and `sin(3*x0)*cos(2*x1)+0.2*x1^2`. The phase target is `theta_next=theta+0.4+0.3*sin(theta)`. The numeric circular control shares exactly the same angle bins and feature expansion as hue. This determines whether any advantage comes from circular features or their color labels.

Image inputs use pixels divided by 255 and no augmentation. They are not synthetic stand-ins. The pilot reuses official test subsets for epoch evaluation, so these results are exploratory validation results rather than an untouched final test claim. Language splits the actual corpus by position before sampling windows, avoiding training windows from the held-out corpus tail.

Precision measurements compare bounded codecs on the same clipped input samples and record how many originals lay outside the domain. Learned scalar centroids fit an independent normal distribution and evaluate all four distributions. Vocabulary sizes are 16, 64, 256, 1,024, 4,096, and 65,536. Uniform codebooks at the same nominal sizes expose whether learned allocation, rather than naming, improves reconstruction.

Weight-only, activation-only, combined quantization, and learned latent variants test different intervention points. Payload-bit corruption and adjacent-token probes separate numerical grid behavior from palette behavior. Adaptive depth measures active token counts and dense storage separately. The learned operator evaluates all finite pairs, so its reported accuracy measures fit to the finite function and does not imply unseen-pair generalization.

MNIST comes through [torchvision's documented dataset loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html). CIFAR-10 comes from [the authors' dataset distribution](https://www.cs.toronto.edu/~kriz/cifar.html). Language uses [Tiny Shakespeare in the char-rnn repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare). The JSON run stores hashes of selected observations and the downloaded corpus.

The methods draw on [PyTorch quantization-aware training](https://pytorch.org/blog/quantization-aware-training/) and [vector-quantized representation learning](https://arxiv.org/abs/1711.00937). The implementation does not claim invention of these methods.
