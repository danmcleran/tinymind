---
title: Home
layout: default
nav_order: 1
---

# Machine Learning for Embedded Systems
{: .fs-9 }

**TinyMind** is a header-only C++ template library for neural networks and machine learning, designed for embedded systems with no FPU, GPU, or OS requirements.
{: .fs-6 .fw-300 }

[Get Started]({{ site.baseurl }}/getting-started/xor-under-4kb){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/danmcleran/tinymind){: .btn .fs-5 .mb-4 .mb-md-0 }
[Example Gallery]({{ site.baseurl }}/gallery){: .btn .fs-5 .mb-4 .mb-md-0 }

![int8 XOR decision surface]({{ site.baseurl }}/assets/plots/xor_decision_surface.png)

*A PyTorch-trained XOR network running pure-integer int8 inference in TinyMind — see the [Example Gallery]({{ site.baseurl }}/gallery) for behavior graphs of every example (each writes a CSV + ships a matplotlib `plot.py`).*

---

## Why TinyMind?

Most ML frameworks assume abundant resources: gigabytes of RAM, a GPU, an operating system, floating-point hardware. TinyMind assumes none of these. It targets the opposite end of the spectrum -- microcontrollers and embedded processors where every byte matters.

| | TinyMind (Q8.8 MLP) | TF Lite Micro | PyTorch |
|---|---|---|---|
| Minimum footprint | 144 bytes (inference) | ~50-100 KB | ~100 MB |
| Trainable XOR network | 328 bytes | N/A (inference only) | ~100 MB |
| Requires FPU | No | Recommended | Yes |
| Requires OS | No | No | Yes |
| Runtime library | None (header-only) | ~50 KB interpreter | ~100 MB |
| GPU required | No | No | Recommended |

Because TinyMind is a header-only template library, there is no runtime library to link. The compiler generates code only for the specific network topology, value type, and activation functions you use. Unused features are never compiled.

Preprocessor capability gates control optional dependencies on the FPU, the C++ stdlib (`<cmath>` / `<type_traits>` / `namespace std`), file I/O, ostreams, `rand()`, the [int8 affine quantization]({{ site.baseurl }}/architectures/int8-quantization) layer family, the [fp16/bf16 storage tier]({{ site.baseurl }}/architectures/mixed-precision), the int16 cell-state variant of `QLSTMCell`, OpenMP outer-loop parallelism, and per-ISA [SIMD specializations]({{ site.baseurl }}/architectures/simd-backends) (NEON / NEON_DOTPROD / NEON_FP16 / SVE / SVE2 / Helium MVE_I / MVE_F / AVX2 / AVX_VNNI / AVX512F / AVX512_VNNI). All default off — a freestanding build pulls in only `<cstddef>` and `<cstdint>` with byte-identical output to the scalar reference. See the [README's Platform Feature Gates](https://github.com/danmcleran/tinymind#platform-feature-gates) section for the full matrix.

## What Fits Where?

TinyMind networks are small enough to deploy on the most constrained microcontrollers:

| Target Device | Flash | RAM | What Fits |
|---|---|---|---|
| ARM Cortex-M0+ (e.g. STM32L0) | 16-64 KB | 4-8 KB | MLP, Elman, Q-learner (inference and training) |
| ARM Cortex-M4 (e.g. STM32L4) | 256 KB-1 MB | 64-128 KB | All architectures including LSTM, GRU, KAN, DQN |
| ARM Cortex-M7 (e.g. STM32H7) | 1-2 MB | 512 KB+ | Multiple networks, full Conv1D pipelines, on-device training |

## Network Architectures

| Type | Class | Notes |
|---|---|---|
| Feed-forward | [`NeuralNetwork`]({{ site.baseurl }}/neural-networks) | Arbitrary depth/width (`MultilayerPerceptron` alias for uniform layers) |
| 1D Convolution | [`Conv1D`]({{ site.baseurl }}/architectures/conv-pooling) | Time-series feature extraction |
| 2D Convolution | [`Conv2D`]({{ site.baseurl }}/architectures/conv-pooling) | Spectrograms, images, time-frequency tiles (NHWC) |
| Depthwise-Separable | [`DepthwiseConv2D` + `PointwiseConv2D`]({{ site.baseurl }}/architectures/conv-pooling) | MobileNet-style block, ~8-9x MAC reduction |
| Max/Avg Pooling | [`MaxPool1D`, `AvgPool1D`]({{ site.baseurl }}/architectures/conv-pooling) | 1D downsampling |
| 2D Pooling | [`MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`]({{ site.baseurl }}/architectures/conv-pooling) | 2D downsampling; GAP replaces flatten-to-dense |
| Binary Dense | [`BinaryDense`]({{ site.baseurl }}/architectures/quantized-networks) | XNOR+popcount (1-bit, 32x compression) |
| Ternary Dense | [`TernaryDense`]({{ site.baseurl }}/architectures/quantized-networks) | Multiply-free ({-1,0,+1}, 16x compression) |
| Int8 Affine | [`QDense`, `QConv2D`, `QDepthwiseConv2D`, ...]({{ site.baseurl }}/architectures/int8-quantization) | TFLite/CMSIS-NN style post-training int8 (per-tensor / per-channel calibration, integer Requantizer). Composition ops: `QConv2DPerChannel`, `QAdd`/`QMul`/`QConcat`/`QPad`, `QBatchNorm`/`QLayerNorm`/`QSoftmax` and `foldBatchNorm` |
| Int8 Recurrent | [`QLSTMCell`, `QGRUCell`, `QCfCCell`]({{ site.baseurl }}/architectures/int8-quantization) | Single-step int8 cells, TFLite gate ordering. Int16 cell-state variant for long unrolls; `QCfCCell` is the closed-form continuous-time (liquid) cell |
| Int8 Attention + FFT | [`QFFT1D`, `QAttention1D`, `QAttentionSoftmax1D`, `QMultiHeadLinearAttention1D`]({{ site.baseurl }}/architectures/int8-quantization) | Q1.15 twiddle FFT, linear and softmax attention, multi-head stack |
| Mixed Precision | [`qbridge`, `fp16_t`, `bf16_t`]({{ site.baseurl }}/architectures/mixed-precision) | Pointwise converters between int8 affine / Q-format / float / fp16 / bf16 |
| SIMD Backends | [NEON / SVE / Helium / AVX2 / AVX-512]({{ site.baseurl }}/architectures/simd-backends) | ISA-capability gates, byte-identical to scalar |
| KAN | [`KolmogorovArnoldNetwork`]({{ site.baseurl }}/architectures/kan) | Learnable B-spline activations |
| FFT | [`FFT1D`]({{ site.baseurl }}/architectures/fft) | Frequency-domain feature extraction for signal processing |
| Elman RNN | [`ElmanNeuralNetwork`]({{ site.baseurl }}/architectures/lstm-gru) | Simple recurrent feedback |
| LSTM | [`LstmNeuralNetwork`]({{ site.baseurl }}/architectures/lstm-gru) | 4-gate architecture |
| GRU | [`GruNeuralNetwork`]({{ site.baseurl }}/architectures/lstm-gru) | 3-gate architecture (~25% less memory) |
| Liquid (LTC / CfC) | [`LtcCell`, `CfCCell`]({{ site.baseurl }}/architectures/lstm-gru#liquid-neural-networks-continuous-time) | Continuous-time cells (fused ODE solver / closed-form). Irregular sampling; train via reverse-mode autodiff; int8 `QCfCCell` for deployment |
| Forward-mode Autodiff (PINN) | [`Dual`, `tanh(Dual)` / `sigmoid(Dual)`]({{ site.baseurl }}/pinn-feasibility) | Input-coordinate derivatives (`du/dx`, `d²u/dx²`) for PDE residuals; float or fixed-point, freestanding-clean |

## Quick Start

```cpp
#include "neuralnet.hpp"

// Q8.8 fixed-point -- no FPU needed
typedef tinymind::QValue<8, 8, true> ValueType;

typedef tinymind::FixedPointTransferFunctions<ValueType,
    RandomNumberGenerator,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>> TransferFunctionsType;

// 2 inputs, 1 hidden layer with 3 neurons, 1 output
typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 3, 1,
    TransferFunctionsType> NeuralNetworkType;

NeuralNetworkType nn;

// Train
nn.feedForward(&inputs[0]);
ValueType error = nn.calculateError(&expected[0]);
if (!TransferFunctionsType::isWithinZeroTolerance(error))
{
    nn.trainNetwork(&expected[0]);
}
```

See the [Neural Network in Under 4KB]({{ site.baseurl }}/getting-started/xor-under-4kb) tutorial for a complete walkthrough.

## Train in PyTorch, Deploy in TinyMind

For many embedded applications, the optimal workflow is:

1. **Train** with PyTorch on a workstation with GPU acceleration
2. **Export** weights to a text file, converting to fixed-point Q-format -- or to int8 with per-tensor / per-channel `(scale, zero_point)` calibration
3. **Deploy** in TinyMind C++ as a non-trainable network at 40-60% less memory, or as a pure-integer int8 pipeline with the [`Q*` layer family]({{ site.baseurl }}/architectures/int8-quantization)

See [PyTorch Interoperability]({{ site.baseurl }}/training/pytorch-interop) for the Q-format flow, [PyTorch → TinyMind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor) for the from-scratch affine int8 walkthrough, and [PyTorch → TinyMind int8 (importer)]({{ site.baseurl }}/getting-started/pytorch-importer) for the production importer flow that consumes a `torch.state_dict` directly and runs calibration with `PercentileObserver` / `KLDivergenceObserver` + cross-layer equalization.

## Activation Function Lookup Tables

Fixed-point activation functions (sigmoid, tanh, exp, log) and trigonometric functions (sin, cos) are implemented via pre-computed lookup tables with linear interpolation -- no FPU or math library needed. Each table is 96 entries, and compile-time preprocessor switches ensure you only pay for the tables you use (96 bytes for a Q8.8 tanh table). See [Activation Function Lookup Tables]({{ site.baseurl }}/activation-luts) for the full details on table generation, runtime lookup, and memory footprint.

## Design Philosophy

TinyMind is inspired by Andrei Alexandrescu's [Modern C++ Design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design). Neural networks are configured through policy classes as template parameters, allowing compile-time specialization that eliminates unused code and data. You only pay for what you use.
