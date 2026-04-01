---
title: Conv & Pooling Layers
layout: default
parent: Architectures
nav_order: 3
---

# Convolutional and Pooling Layers

Tinymind provides standalone composable signal processing layers for time-series feature extraction: `Conv1D`, `MaxPool1D`, `AvgPool1D`, `BatchNorm1D`, and `Dropout`. These layers sit outside the neural network template and can be chained together into a feature extraction pipeline whose output feeds into a standard neural network for classification or regression.

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
