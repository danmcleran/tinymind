---
title: Quantized Networks
layout: default
parent: Architectures
nav_order: 4
---

# Quantized Neural Networks

Tinymind provides two extreme-quantization layer types for ultra-low-power inference: `BinaryDense` and `TernaryDense`. These layers replace full-precision multiply-accumulate operations with bitwise logic (XNOR + popcount for binary) or conditional add/subtract/skip (for ternary), achieving massive memory reduction and eliminating multiplication entirely from the forward pass.

- **BinaryDense**: Weights and activations constrained to {-1, +1}. 32x memory reduction via 1-bit packing.
- **TernaryDense**: Weights constrained to {-1, 0, +1}. 16x memory reduction via 2-bit packing. Zero weights are skipped, providing sparsity.

Both layers support training via the Straight-Through Estimator (STE) and work with both fixed-point and floating-point value types.

## Why Extreme Quantization on Embedded?

On the smallest microcontrollers, even Q8.8 fixed-point may be too large for wide layers. Binary and ternary quantization make previously impossible deployments feasible:

| 64x16 layer weights | `double` | Q8.8 | Binary (1-bit) | Ternary (2-bit) |
|---|---|---|---|---|
| Weight storage | 8,192 bytes | 2,048 bytes | 128 bytes | 256 bytes |

A layer that would consume 8 KB in full precision fits in 128 bytes with binary packing -- small enough for an ARM Cortex-M0+ with 4 KB of RAM to run multiple layers simultaneously.

# BinaryDense

## Template Declaration

```cpp
template<
    typename ValueType,
    size_t InputSize,
    size_t OutputSize>
class BinaryDense
```

## How It Works

### Forward Pass
1. **Binarize inputs**: `sign(x)` maps each input to +1 or -1
2. **Pack**: Both inputs and weights are stored as single bits
3. **XNOR**: Bitwise XNOR gives 1 where input and weight have the same sign
4. **Popcount**: Count the set bits to get the number of agreements
5. **Dot product**: `output = 2 * popcount(XNOR(input, weight)) - InputSize + bias`

No multiplication is performed.

### Training (Straight-Through Estimator)
During training, real-valued "latent" weights are maintained alongside the packed binary weights. The STE passes gradients through the `sign()` binarization as if it were the identity function, with gradients clipped to zero for latent weights outside [-1, +1].

## Example

```cpp
#include "binarylayer.hpp"

tinymind::BinaryDense<double, 4, 2> layer;

// Set latent weights
layer.setLatentWeight(0, 0,  0.5);  // binarizes to +1
layer.setLatentWeight(0, 1,  0.3);  // binarizes to +1
layer.setLatentWeight(0, 2, -0.7);  // binarizes to -1
layer.setLatentWeight(0, 3, -0.2);  // binarizes to -1
layer.setBias(0, 0.0);

layer.binarizeWeights(); // pack into bits

double input[4] = {1.0, -1.0, 1.0, -1.0};
double output[2];
layer.forward(input, output);
```

# TernaryDense

## Template Declaration

```cpp
template<
    typename ValueType,
    size_t InputSize,
    size_t OutputSize,
    unsigned ThresholdPercent = 50>
class TernaryDense
```

## How It Works

### Ternarization
Weights are quantized to {-1, 0, +1} based on a threshold:
1. Compute the mean absolute weight: `mean_abs = mean(|w|)`
2. Apply threshold: `threshold = ThresholdPercent/100 * mean_abs`
3. For each weight: if `|w| < threshold` -> 0, else `sign(w)` -> +1 or -1

### Forward Pass
For each output neuron: weight = +1: add input; weight = -1: subtract input; weight = 0: skip.

No multiplication is performed. Zero weights provide natural sparsity.

## Example

```cpp
#include "ternarylayer.hpp"

tinymind::TernaryDense<double, 4, 2, 50> layer;

layer.setLatentWeight(0, 0,  0.9);   // -> +1
layer.setLatentWeight(0, 1,  0.01);  // -> 0 (pruned)
layer.setLatentWeight(0, 2, -0.8);   // -> -1
layer.setLatentWeight(0, 3,  0.02);  // -> 0 (pruned)
layer.setBias(0, 0.0);

layer.ternarizeWeights();

double input[4] = {2.0, 3.0, 4.0, 5.0};
double output[2];
layer.forward(input, output);
// output[0] = (+1)*2 + 0*3 + (-1)*4 + 0*5 = -2.0
```

# Fixed-Point Support

Both layers work with Q-format fixed-point types:

```cpp
typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;

tinymind::BinaryDense<ValueType, 4, 1> binaryLayer;
tinymind::TernaryDense<ValueType, 4, 1, 10> ternaryLayer;
```

# Compression Ratios (64x16 layer)

| Storage | Bytes | vs `double` | vs Q8.8 |
|---|---|---|---|
| Full `double` | 8,192 bytes | 1x | -- |
| Full Q8.8 | 2,048 bytes | 4x | 1x |
| Packed binary (1-bit) | 128 bytes | **64x** | **16x** |
| Packed ternary (2-bit) | 256 bytes | **32x** | **8x** |

# When To Use Binary vs Ternary

- **BinaryDense**: Maximum compression (32x). Best when the problem is simple enough that {-1, +1} weights suffice. XNOR+popcount is extremely fast on hardware with popcount instructions.
- **TernaryDense**: Slightly less compression (16x) but supports pruning via zero weights. The sparsity can skip operations entirely, and the ability to "turn off" connections gives more expressiveness than pure binary.

Both are best suited for the later (wider) layers of a network where weight storage dominates memory.
