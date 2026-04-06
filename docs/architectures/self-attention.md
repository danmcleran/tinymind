---
title: Linear Self-Attention
layout: default
parent: Architectures
nav_order: 5
---

# Linear Self-Attention

Tinymind provides `SelfAttention1D`, a standalone composable layer that implements linear self-attention for sequence processing. It uses a ReLU kernel feature map instead of softmax, reducing complexity from O(N^2 * D) to O(N * D * P + N * P^2) and eliminating the need for exponential or division operations -- making it practical for fixed-point targets.

## Why Linear Attention?

Standard self-attention computes:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
```

The `Q * K^T` product creates an N x N matrix, which is O(N^2) in both compute and memory. On an MCU with kilobytes of SRAM, this is infeasible for any non-trivial sequence length.

Linear attention replaces softmax with a kernel feature map `phi()` and reorders the computation:

```
Standard:  (phi(Q) * phi(K)^T) * V    -- O(N^2 * P) -- N x N intermediate
Linear:    phi(Q) * (phi(K)^T * V)    -- O(N * P^2) -- P x P intermediate
```

By computing `K^T * V` first (a P x P matrix, where P = ProjectionDim), the N x N matrix is never formed. When P << N, this is a massive reduction.

Tinymind uses `phi(x) = ReLU(x)`, which:
- Is already a native activation function in the library
- Requires no lookup tables, no exp, no division
- Works identically for both float and Q-format fixed-point types
- Guarantees non-negative keys and queries (required for valid attention kernels)

## Template Declaration

```cpp
template<
    typename ValueType,
    size_t SequenceLength,
    size_t EmbeddingDim,
    size_t ProjectionDim>
class SelfAttention1D
```

**Template Parameters:**
- `ValueType` -- Numeric type (`QValue`, `float`, or `double`)
- `SequenceLength` -- Number of input time steps (N)
- `EmbeddingDim` -- Input feature dimension per time step (D)
- `ProjectionDim` -- Dimension of Q, K, V projections (P)

**Static Properties:**
- `InputSize = SequenceLength * EmbeddingDim`
- `OutputSize = SequenceLength * ProjectionDim`
- `WeightsPerProjection = EmbeddingDim * ProjectionDim`
- `TotalWeights = 3 * WeightsPerProjection` (W_q, W_k, W_v)
- `TotalBiases = 3 * ProjectionDim` (b_q, b_k, b_v)

## Forward Pass

The forward pass performs five steps:

```
Q' = ReLU(X * W_q + b_q)     N x P    -- query projection
K' = ReLU(X * W_k + b_k)     N x P    -- key projection
V  = X * W_v + b_v            N x P    -- value projection (no activation)
KV = K'^T * V                 P x P    -- associative product
Out = Q' * KV                 N x P    -- attended output
```

All operations are matrix multiplications and ReLU -- both of which Q-format handles natively. There is no softmax, no exponential, and no division in the forward pass.

## Example: Floating-Point

```cpp
#include "selfattention1d.hpp"

// 4 time steps, 2-dim embedding, 2-dim projections
tinymind::SelfAttention1D<double, 4, 2, 2> attn;

// Set W_q = W_k = W_v = identity
for (size_t proj = 0; proj < 3; ++proj)
{
    attn.setProjectionWeight(proj, 0, 0, 1.0);
    attn.setProjectionWeight(proj, 0, 1, 0.0);
    attn.setProjectionWeight(proj, 1, 0, 0.0);
    attn.setProjectionWeight(proj, 1, 1, 1.0);
}

double input[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
double output[8];

attn.forward(input, output);
// With identity projections on positive input:
// Q' = K' = V = X, KV = X^T * X, Out = X * (X^T * X)
```

## Example: Fixed-Point (Q16.16)

```cpp
#include "selfattention1d.hpp"

typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
tinymind::SelfAttention1D<ValueType, 4, 2, 2> attn;

// Set identity projections
for (size_t proj = 0; proj < 3; ++proj)
{
    attn.setProjectionWeight(proj, 0, 0, ValueType(1, 0));
    attn.setProjectionWeight(proj, 0, 1, ValueType(0));
    attn.setProjectionWeight(proj, 1, 0, ValueType(0));
    attn.setProjectionWeight(proj, 1, 1, ValueType(1, 0));
}

ValueType input[8];
input[0] = ValueType(1, 0); input[1] = ValueType(2, 0);
input[2] = ValueType(3, 0); input[3] = ValueType(4, 0);
input[4] = ValueType(5, 0); input[5] = ValueType(6, 0);
input[6] = ValueType(7, 0); input[7] = ValueType(8, 0);

ValueType output[8];
attn.forward(input, output);
```

## Pipeline: Conv1D + Self-Attention

Self-attention pairs naturally with Conv1D for sensor processing pipelines where Conv1D extracts local features and self-attention models temporal dependencies:

```cpp
#include "conv1d.hpp"
#include "selfattention1d.hpp"
#include "neuralnet.hpp"

// Conv1D: extract features from 128-point sensor input
typedef tinymind::Conv1D<double, 128, 5, 2, 4> ConvType;
// Output: 62 positions x 4 filters = 248 values

// Self-attention: model dependencies across 62 time steps
typedef tinymind::SelfAttention1D<double, 62, 4, 4> AttnType;
// Output: 62 x 4 = 248 values

ConvType conv;
AttnType attn;

double sensorData[128];
double convOut[ConvType::OutputSize];
double attnInput[62 * 4]; // reshaped from filter-major to time-step-major
double attnOut[AttnType::OutputSize];

conv.forward(sensorData, convOut);

// Reshape conv output (filter-major) to attention input (time-step-major)
for (size_t t = 0; t < 62; ++t)
    for (size_t f = 0; f < 4; ++f)
        attnInput[t * 4 + f] = convOut[f * 62 + t];

attn.forward(attnInput, attnOut);

// Feed into classifier
classifier.feedForward(attnOut);
```

## Backward Pass and Training

`SelfAttention1D` supports full backpropagation via `computeGradients()` and SGD weight updates:

```cpp
// After forward pass
double outputDeltas[AttnType::OutputSize]; // from downstream layer

attn.computeGradients(outputDeltas);
attn.updateWeights(-0.01); // negative learning rate for gradient descent
```

The backward pass computes gradients for all three projection matrices (W_q, W_k, W_v) and their biases by backpropagating through the linear attention computation.

## Weight Accessors

Weights can be accessed by flat index or by projection/row/column:

```cpp
// Structured access: projection (0=Q, 1=K, 2=V), row, column
attn.setProjectionWeight(0, row, col, value);  // W_q
attn.setProjectionWeight(1, row, col, value);  // W_k
attn.setProjectionWeight(2, row, col, value);  // W_v

// Bias access: projection, index
attn.setProjectionBias(0, idx, value);  // b_q
attn.setProjectionBias(1, idx, value);  // b_k
attn.setProjectionBias(2, idx, value);  // b_v

// Flat access (for serialization)
attn.setWeight(flatIndex, value);
attn.setBias(flatIndex, value);
```

## Memory Footprint

All storage is statically allocated. The total footprint includes weights, gradients, biases, and cached intermediate values for backpropagation:

| Component | Count | Purpose |
|---|---|---|
| W_q, W_k, W_v | 3 * D * P | Projection weights |
| Gradients | 3 * D * P | Weight gradients |
| b_q, b_k, b_v | 3 * P | Projection biases |
| Bias gradients | 3 * P | Bias gradients |
| Input cache | N * D | Cached for backward pass |
| Q', K', V | 3 * N * P | Projection outputs |
| KV | P * P | Associative product |
| Output cache | N * P | Cached for backward pass |

**Total values = 6*D*P + 6*P + N*D + 4*N*P + P^2**

### Recommended Configurations

| Target | ValueType | N | D | P | Weights | Total RAM |
|---|---|---|---|---|---|---|
| Small MCU (Cortex-M0, 8-32KB) | Q8.8 | 16 | 8 | 4 | 96 | ~768 bytes |
| Mid-range MCU (Cortex-M4, 64-256KB) | Q16.16 | 32 | 16 | 8 | 384 | ~6 KB |
| Large MCU (Cortex-M7, 256KB+) | Q16.16 | 64 | 32 | 16 | 1,536 | ~32 KB |

## Q-Format Considerations

### Accumulator Width

Each dot product in the matrix multiplications accumulates `P` multiply-accumulate operations. Each Q-format multiply doubles the bit width before the rounding policy truncates back:

| ValueType | Safe ProjectionDim | Reason |
|---|---|---|
| Q8.8 | up to 4 | Accumulator fits in 32-bit with 2 bits headroom |
| Q16.16 | up to 8 | Accumulator fits in 128-bit (via `typeChooser.hpp`) |
| Q24.8 | up to 8 | Same 128-bit accumulator |

**Q16.16 with ProjectionDim=8 is the recommended sweet spot** -- enough precision for meaningful attention patterns while keeping accumulator math clean.

### Why No Softmax?

Softmax requires:
1. Exponential function (exp) for each element
2. Sum across the sequence
3. Division to normalize

In fixed-point, all three are expensive and introduce quantization noise. Linear attention with ReLU avoids all of them -- the entire forward pass is multiply-accumulate plus ReLU comparison, both of which Q-format handles natively with no lookup tables.

## Performance Estimates

### ARM Cortex-A53 (1.2 GHz)

Mid-range config (Q16.16, N=32, D=16, P=8), 16,384 total MACs:

| Scenario | Estimate |
|---|---|
| Scalar, -O3, WrapPolicy | ~48 us |
| Scalar, -O3, MinMaxSaturatePolicy | ~75 us |
| Auto-vectorized inner loops | ~30 us |

### ARM Cortex-M0 (48 MHz)

| Config | Estimate |
|---|---|
| Q8.8, N=16, D=8, P=4, 1-cycle MUL | ~170 us |
| Q16.16, N=32, D=16, P=8, 1-cycle MUL | ~8.5 ms |

The Q8.8 small config is real-time capable at 100-1000 Hz sensor rates on M0. The Q16.16 mid-range config is better suited for M3/M4 or higher for real-time use.

## Use Cases

- **Sensor fusion** -- attend across multiple IMU/accelerometer channels to learn per-timestep channel importance
- **Anomaly detection** -- attend across a time window of sensor readings to identify unusual patterns
- **Keyword spotting** -- attend across audio feature frames (after Conv1D feature extraction)
- **Sequence classification** -- replace or augment GRU for fixed-length sequence tasks with lower latency (no sequential dependency in the forward pass)
