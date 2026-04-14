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
| Max/Avg Pooling | [`MaxPool1D`, `AvgPool1D`]({{ site.baseurl }}/architectures/conv-pooling) | Downsampling |
| Binary Dense | [`BinaryDense`]({{ site.baseurl }}/architectures/quantized-networks) | XNOR+popcount (1-bit, 32x compression) |
| Ternary Dense | [`TernaryDense`]({{ site.baseurl }}/architectures/quantized-networks) | Multiply-free ({-1,0,+1}, 16x compression) |
| KAN | [`KolmogorovArnoldNetwork`]({{ site.baseurl }}/architectures/kan) | Learnable B-spline activations |
| FFT | [`FFT1D`]({{ site.baseurl }}/architectures/fft) | Frequency-domain feature extraction for signal processing |
| Elman RNN | [`ElmanNeuralNetwork`]({{ site.baseurl }}/architectures/lstm-gru) | Simple recurrent feedback |
| LSTM | [`LstmNeuralNetwork`]({{ site.baseurl }}/architectures/lstm-gru) | 4-gate architecture |
| GRU | [`GruNeuralNetwork`]({{ site.baseurl }}/architectures/lstm-gru) | 3-gate architecture (~25% less memory) |

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
2. **Export** weights to a text file, converting to fixed-point Q-format
3. **Deploy** in TinyMind C++ as a non-trainable network at 40-60% less memory

See [PyTorch Interoperability]({{ site.baseurl }}/training/pytorch-interop) for details.

## Activation Function Lookup Tables

Fixed-point activation functions (sigmoid, tanh, exp, log) and trigonometric functions (sin, cos) are implemented via pre-computed lookup tables with linear interpolation -- no FPU or math library needed. Each table is 96 entries, and compile-time preprocessor switches ensure you only pay for the tables you use (96 bytes for a Q8.8 tanh table). See [Activation Function Lookup Tables]({{ site.baseurl }}/activation-luts) for the full details on table generation, runtime lookup, and memory footprint.

## Design Philosophy

TinyMind is inspired by Andrei Alexandrescu's [Modern C++ Design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design). Neural networks are configured through policy classes as template parameters, allowing compile-time specialization that eliminates unused code and data. You only pay for what you use.
