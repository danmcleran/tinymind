---
title: Conv & Pooling Layers
layout: default
parent: Architectures
nav_order: 3
---

# Convolutional and Pooling Layers

Tinymind provides standalone composable signal processing layers for both 1D time-series and 2D spatial feature extraction:

- **1D:** `Conv1D`, `MaxPool1D`, `AvgPool1D`, `BatchNorm1D`, `Dropout`
- **2D:** `Conv2D`, `DepthwiseConv2D`, `PointwiseConv2D`, `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`

These layers sit outside the neural network template and can be chained together into a feature extraction pipeline whose output feeds into a standard neural network for classification or regression.

These layers are particularly useful for embedded sensor applications where raw time-series data needs to be transformed into discriminative features before feeding into a small classifier network:

- **Accelerometer/IMU** -- gesture recognition, fall detection, activity classification on wearables
- **ECG/PPG** -- arrhythmia detection, heart rate variability analysis on medical devices
- **Vibration sensors** -- bearing fault detection, motor health monitoring on industrial edge nodes
- **Microphone** -- keyword spotting, anomaly detection on always-on audio devices

A complete Conv1D -> MaxPool1D -> Dropout pipeline in Q8.8 fixed-point takes 1,825 bytes -- feasible on any ARM Cortex-M class device.

# Conv1D - 1D Convolution Layer

Conv1D applies a set of learned filters (kernels) to a 1D input signal, producing a multi-channel feature map.

## Template Declaration

```cpp
template<
    typename ValueType,
    size_t InputLength,
    size_t KernelSize,
    size_t Stride = 1,
    size_t NumFilters = 1>
class Conv1D
```

**Template Parameters:**
- `ValueType` - Numeric type (`QValue`, `float`, or `double`)
- `InputLength` - Number of input time steps
- `KernelSize` - Convolution kernel width
- `Stride` - Step size between kernel positions (default 1)
- `NumFilters` - Number of output feature channels (default 1)

**Static Properties:**
- `OutputLength = (InputLength - KernelSize) / Stride + 1`
- `OutputSize = NumFilters * OutputLength`

## Example

```cpp
#include "conv1d.hpp"

// 5-point input, kernel size 3, stride 1, 1 filter
tinymind::Conv1D<double, 5, 3, 1, 1> conv;

// Set kernel weights: [1, 0, -1] with bias 0 (difference filter)
conv.setFilterWeight(0, 0, 1.0);
conv.setFilterWeight(0, 1, 0.0);
conv.setFilterWeight(0, 2, -1.0);
conv.setFilterWeight(0, 3, 0.0); // bias

double input[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
double output[3]; // OutputLength = (5 - 3) / 1 + 1 = 3

conv.forward(input, output);
// output[0] = -2.0, output[1] = -2.0, output[2] = -2.0
```

# MaxPool1D - 1D Max Pooling Layer

MaxPool1D downsamples by selecting the maximum value within each pooling window. It tracks argmax indices for gradient routing during backpropagation.

```cpp
template<
    typename ValueType,
    size_t InputLength,
    size_t PoolSize,
    size_t Stride = PoolSize,
    size_t NumChannels = 1>
class MaxPool1D
```

## Example

```cpp
#include "pool1d.hpp"

tinymind::MaxPool1D<double, 6, 2, 2, 1> pool;
double input[6] = {1.0, 3.0, 2.0, 5.0, 4.0, 6.0};
double output[3];

pool.forward(input, output);
// output[0] = 3.0, output[1] = 5.0, output[2] = 6.0
```

**Memory Note:** MaxPool1D stores argmax indices for backpropagation. For inference-only deployments, consider `AvgPool1D` which is stateless (1 byte).

# AvgPool1D - 1D Average Pooling Layer

AvgPool1D downsamples by computing the mean within each pooling window. It is **stateless** (1 byte) -- no per-instance storage needed.

```cpp
tinymind::AvgPool1D<double, 6, 2, 2, 1> pool;
double input[6] = {1.0, 3.0, 2.0, 6.0, 4.0, 8.0};
double output[3];

pool.forward(input, output);
// output[0] = 2.0, output[1] = 4.0, output[2] = 6.0
```

# BatchNorm1D - 1D Batch Normalization

BatchNorm1D normalizes each feature to zero mean and unit variance, then applies learnable affine parameters (gamma, beta).

```cpp
template<
    typename ValueType,
    size_t Size,
    unsigned MomentumPercent = 10,
    unsigned EpsilonPercent = 1>
class BatchNorm1D
```

## Training vs Inference

- **Training mode:** Computes mean/variance from the current input, updates running statistics
- **Inference mode:** Uses stored running mean/variance for stable, deterministic normalization

```cpp
#include "batchnorm.hpp"

tinymind::BatchNorm1D<double, 4> bn;
bn.setTraining(true);

double input[4] = {2.0, 4.0, 6.0, 8.0};
double output[4];
bn.forward(input, output);

// Switch to inference mode
bn.setTraining(false);
bn.forward(input, output);
```

# Dropout - Inverted Dropout Regularization

Dropout randomly zeros elements during training to prevent overfitting. Tinymind uses inverted dropout: surviving elements are scaled by `1/(1-p)` during training so that no scaling is needed at inference time.

```cpp
template<
    typename ValueType,
    size_t Size,
    unsigned DropoutPercent = 50>
class Dropout
```

```cpp
#include "dropout.hpp"

tinymind::Dropout<double, 100, 50> dropout; // 50% dropout

// Training mode (default)
dropout.forward(input, output);
// ~50 outputs are 0.0, ~50 outputs are 2.0 (scaled)

// Switch to inference
dropout.setTraining(false);
dropout.forward(input, output);
// All outputs equal input (identity pass-through)
```

**Memory:** `Size + 1` bytes total, independent of value type.

# Full Pipeline Example

Conv1D -> MaxPool1D -> Dropout -> Neural Network:

```cpp
#include "conv1d.hpp"
#include "pool1d.hpp"
#include "dropout.hpp"
#include "neuralnet.hpp"

typedef tinymind::Conv1D<double, 100, 5, 1, 4> ConvType;
typedef tinymind::MaxPool1D<double, ConvType::OutputLength, 2, 2, 4> PoolType;
typedef tinymind::Dropout<double, PoolType::OutputSize, 50> DropoutType;

ConvType conv;
PoolType pool;
DropoutType dropout;

double sensorData[100];
double convOut[ConvType::OutputSize];
double poolOut[PoolType::OutputSize];
double dropOut[PoolType::OutputSize];

// Forward pipeline
conv.forward(sensorData, convOut);
pool.forward(convOut, poolOut);
dropout.forward(poolOut, dropOut);

// Feed into classifier
classifier.feedForward(dropOut);

// Switch to inference (disable dropout)
dropout.setTraining(false);
```

# Pipeline Size

| Value Type | Conv1D (100, k=5, s=1, 4f) | MaxPool1D (96, p=2, s=2, 4ch) | Dropout (192, 50%) | Total |
|---|---|---|---|---|
| `double` | 384 bytes | 1,536 bytes | 193 bytes | **2,113 bytes** |
| Q8.8 | 96 bytes | 1,536 bytes | 193 bytes | **1,825 bytes** |

MaxPool1D dominates pipeline memory due to argmax index storage. For memory-constrained deployments, AvgPool1D eliminates this overhead entirely (1 byte vs 1,536 bytes).

---

# 2D Layers for Spectrograms, Images, and Time-Frequency Tiles

The 2D layers target the same embedded workloads that dominate the TinyML benchmark suite: keyword spotting on MFCC spectrograms, small-image person detection, and 2D vibration / time-frequency anomaly detection.

All 2D layers use **NHWC** (channel-last) layout -- the same ordering used by CMSIS-NN and TFLite Micro -- and VALID padding. Dimensions are compile-time template parameters; no dynamic allocation.

## Conv2D - 2D Convolution Layer

```cpp
template<
    typename ValueType,
    size_t H,             // input height
    size_t W,             // input width
    size_t InChannels,    // input channel count
    size_t KH,            // kernel height
    size_t KW,            // kernel width
    size_t StrideH = 1,
    size_t StrideW = 1,
    size_t NumFilters = 1>
class Conv2D
```

**Static properties:**
- `OutputHeight = (H - KH) / StrideH + 1`
- `OutputWidth  = (W - KW) / StrideW + 1`
- `OutputSize   = OutputHeight * OutputWidth * NumFilters`
- `WeightsPerFilter = KH * KW * InChannels + 1` (bias included)

### Example

```cpp
#include "conv2d.hpp"

// 3x3 input, 1 channel, 3x3 kernel, 1 filter -> 1x1 output.
// Kernel is an identity: only the center tap is non-zero.
tinymind::Conv2D<double, 3, 3, 1, 3, 3, 1, 1, 1> conv;
conv.setFilterWeight(0, 1, 1, 0, 1.0); // (filter, kh, kw, inChannel)
conv.setFilterBias(0, 0.0);

double input[9]  = {1, 2, 3, 4, 5, 6, 7, 8, 9};
double output[1];
conv.forward(input, output);
// output[0] == 5.0 (picks the center pixel)
```

## DepthwiseConv2D - Depthwise Separable Building Block

`DepthwiseConv2D` applies one kernel per input channel with no cross-channel mixing. Output channel count equals input channel count. This is the left half of the MobileNet-style depthwise-separable convolution -- typically paired with `PointwiseConv2D` (1x1) to recover channel mixing at a fraction of the MACs.

For a common shape (K=3, Cin=Cout=32), a depthwise-separable block uses **~8-9x fewer MACs** than a full `Conv2D` block of the same input/output shape.

```cpp
template<
    typename ValueType,
    size_t H, size_t W,
    size_t Channels,      // input == output channel count
    size_t KH, size_t KW,
    size_t StrideH = 1,
    size_t StrideW = 1>
class DepthwiseConv2D
```

## PointwiseConv2D - 1x1 Convolution / Dense Layer

`PointwiseConv2D` is the right half of the separable block: a 1x1 convolution that projects `InChannels` to `NumFilters` at each spatial position. It is also the most convenient way to build a final dense classifier on top of `GlobalAvgPool2D` (1x1 spatial input).

```cpp
template<
    typename ValueType,
    size_t H, size_t W,
    size_t InChannels,
    size_t NumFilters>
class PointwiseConv2D
```

## 2D Pooling

```cpp
template<typename ValueType, size_t H, size_t W, size_t Channels,
         size_t PoolH, size_t PoolW,
         size_t StrideH = PoolH, size_t StrideW = PoolW>
class MaxPool2D;

template</* same parameters */>
class AvgPool2D;

template<typename ValueType, size_t H, size_t W, size_t Channels>
class GlobalAvgPool2D; // Collapses H*W into a single value per channel.
```

`GlobalAvgPool2D` deserves special attention: it replaces the big flatten-to-dense matrix that dominates flash in small CNNs. For a typical 7x7x16 final feature map, flatten + dense to 10 classes is 7840 weights; GAP + 1x1 conv is 170. The dense-layer flash collapses by ~45x.

## Depthwise-Separable Pipeline

A complete MobileNet-style pipeline using the new layers:

```cpp
#include "conv2d.hpp"
#include "depthwiseconv2d.hpp"
#include "pointwiseconv2d.hpp"
#include "pool2d.hpp"

// 20x20x1 -> 10 class logits
using Conv1 = tinymind::Conv2D<float, 20, 20, 1, 3, 3, 1, 1, 8>;   // -> 18x18x8
using Pool1 = tinymind::MaxPool2D<float, 18, 18, 8, 2, 2>;          // -> 9x9x8
using Dw    = tinymind::DepthwiseConv2D<float, 9, 9, 8, 3, 3>;      // -> 7x7x8
using Pw    = tinymind::PointwiseConv2D<float, 7, 7, 8, 16>;        // -> 7x7x16
using Gap   = tinymind::GlobalAvgPool2D<float, 7, 7, 16>;           // -> 16
using Dense = tinymind::PointwiseConv2D<float, 1, 1, 16, 10>;       // -> 10

Conv1 conv1; Pool1 pool1; Dw dw; Pw pw; Gap gap; Dense dense;
float input[400], b1[Conv1::OutputSize], b2[Pool1::OutputSize];
float b3[Dw::OutputSize], b4[Pw::OutputSize], b5[Gap::OutputSize];
float logits[Dense::OutputSize];

conv1.forward(input, b1);
pool1.forward(b1, b2);
dw.forward(b2, b3);
pw.forward(b3, b4);
gap.forward(b4, b5);
dense.forward(b5, logits);
```

See the runnable [`examples/kws_cortex_m/`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m) example, which runs this exact pipeline and prints a per-layer cycle-count + byte-footprint CSV.

### Reference footprint (float32, uninstrumented debug sizes)

| Layer | sizeof(layer) | Activation bytes |
|---|---|---|
| `Conv2D 3x3 1->8` | 640 | 10,368 |
| `MaxPool2D 2x2` | 5,184 | 2,592 |
| `DepthwiseConv2D 3x3 8` | 640 | 1,568 |
| `PointwiseConv2D 8->16` | 1,152 | 3,136 |
| `GlobalAvgPool2D` | 1 | 64 |
| `PointwiseConv2D 16->10` (dense) | 1,360 | 40 |

Total weight bytes: **8,976**. Peak activation: **10,368** bytes (first conv dominates). The layer objects include both weights and gradients -- for pure-inference MCU builds, a planned templated variant drops the gradient array and roughly halves the weight footprint.

---

# Benchmark Harness (`cpp/include/bench/`)

The `bench/` headers ship alongside the 2D layers so you can instrument any TinyMind pipeline with per-layer cycle counts and a RAM high-water mark without pulling in heavyweight dependencies.

**`bench/platform.hpp`**
- `bench::readCycleCounter()` -- reads `DWT->CYCCNT` when built with `-DTINYMIND_BENCH_CORTEX_M`, falls back to `std::chrono::steady_clock` nanoseconds on the host.
- `bench::enableCycleCounter()` -- one-time DWT init on Cortex-M targets that expose it (M3/M4/M7/M33/M55).
- `bench::paintStack(buffer, size)` + `bench::stackHighWater(buffer, size)` -- canary-based stack watermarking for measuring worst-case RAM on MCUs.

**`bench/report.hpp`**
- `bench::LayerStat` -- a 4-field struct (`name`, `weight_bytes`, `activation_bytes`, `cycles`).
- `bench::writeHeader(sink)` / `bench::writeRow(sink, row)` -- CSV output against any sink that implements `operator<<(const char*)` and `operator<<(size_t/uint32_t)`. Works with `std::ostream` on host and with a lightweight UART wrapper on an MCU -- no `<iostream>` dependency required.
- `bench::ScopedTimer` -- RAII cycle timer; call `readElapsed()` for cycles since construction.

### Usage

```cpp
#include "bench/platform.hpp"
#include "bench/report.hpp"

tinymind::bench::enableCycleCounter();
tinymind::bench::writeHeader(std::cout);

const auto t0 = tinymind::bench::readCycleCounter();
conv1.forward(input, b1);
const auto t1 = tinymind::bench::readCycleCounter();
// ...

tinymind::bench::writeRow(std::cout,
    {"conv2d_3x3_8", sizeof(conv1), sizeof(b1), t1 - t0});
```

The [`examples/kws_cortex_m/`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m) example ties the harness to a full keyword-spotting-style pipeline and ships a `port_stub.hpp` with a minimal UART sink for Cortex-M porting.
