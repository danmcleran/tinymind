# TinyMind

A header-only C++ template library for neural networks, Kolmogorov-Arnold Networks (KAN), LSTM and GRU recurrent networks, linear self-attention, FFT-based signal processing, binary and ternary neural networks, and Q-learning, designed for embedded systems with no FPU, GPU, or vectorized instruction requirements.

Inspired by Andrei Alexandrescu's policy-based design from [Modern C++ Design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design), TinyMind uses template metaprogramming to produce zero-overhead abstractions where network topology, value type, activation functions, and training policies are all compile-time parameters.

## Features

### Neural Networks

- **Feed-forward networks** with arbitrary depth and width
- **1D convolution layer** for time-series feature extraction (sensor data, IMU, ECG)
- **1D pooling layers** (`MaxPool1D`, `AvgPool1D`) for downsampling with multi-channel support and backpropagation
- **Linear self-attention** (`SelfAttention1D`) using ReLU kernel feature map -- O(N*P^2) instead of O(N^2*D), no softmax/exp required, works with Q-format fixed-point
- **FFT layer** (`FFT1D`) with radix-2 decimation-in-time, compile-time bit-reversal tables, and scaled butterfly stages for fixed-point overflow prevention -- frequency-domain feature extraction for signal processing pipelines
- **Kolmogorov-Arnold Networks (KAN)** with learnable B-spline activation functions on edges
- **Recurrent neural networks** (Elman) with configurable recurrent connection depth
- **LSTM networks** with gated cell state, supporting single and multi-layer configurations
- **GRU networks** (Gated Recurrent Unit) with 3-gate architecture -- ~25% less memory than LSTM per hidden neuron
- **Binary neural networks** (`BinaryDense`) with XNOR+popcount forward pass -- 32x weight memory reduction via bit packing, no multiplication needed
- **Ternary neural networks** (`TernaryDense`) with {-1, 0, +1} weights -- 16x weight memory reduction via 2-bit packing, multiply-free forward pass using conditional add/subtract/skip
- **Heterogeneous hidden layers** via `HiddenLayers<N0, N1, ...>` for different neuron counts per layer (`MultilayerPerceptron` is a convenience alias for uniform layers)
- **Batch training** with configurable batch size
- **Softmax output** for multi-class classification
- **Xavier weight initialization** (uniform and normal distributions)
- **Weight import/export** in CSV and binary formats for all network types (MLP, LSTM, GRU, KAN; interoperable with PyTorch)

### Training Policies

- **Adam optimizer** with per-parameter adaptive learning rates (`AdamOptimizerFloat` for double, `AdamOptimizer` for fixed-point)
- **RMSprop optimizer** with per-parameter adaptive learning rates via running average of squared gradients (`RmsPropOptimizerFloat` for double, `RmsPropOptimizer` for fixed-point) -- often preferred over Adam for recurrent networks
- **Dropout regularization** via `Dropout<ValueType, Size, DropoutPercent>` -- inverted dropout with training/inference mode toggle, no scaling needed at inference time
- **Gradient clipping** via configurable `GradientClipByValue` policy to prevent exploding gradients (especially critical for fixed-point)
- **L2 weight decay** (ridge regularization) via `L2WeightDecay` policy to prevent weight overflow and reduce overfitting
- **Learning rate scheduling** with `StepDecaySchedule` (multiply by decay factor every N steps) and `FixedLearningRatePolicy` (default)
- **Early stopping** via `EarlyStopping<ValueType, Patience>` to detect convergence and save compute cycles
- **Teacher forcing / scheduled sampling** via `ScheduledSampling` for recurrent training -- linearly decays from ground truth to model predictions
- **Truncated BPTT** via `TruncatedBPTT<NNType, WindowSize>` -- accumulates recurrent state over K timesteps before weight update
- All policies are optional template parameters with null/no-op defaults for full backward compatibility
- Policies are extracted from `TransferFunctionsPolicy` via SFINAE traits -- existing user code compiles unchanged

### Kolmogorov-Arnold Networks (KAN)

- Based on the [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)
- Learnable B-spline activation functions on edges, pure summation at nodes
- Edge function: `phi(x) = w_b * SiLU(x) + w_s * spline(x)`
- Configurable spline degree (`SplineDegree`) and grid resolution (`GridSize`)
- Piecewise linear specialization (`SplineDegree=1`) for fixed-point targets
- SiLU activation reuses existing sigmoid lookup tables -- no new tables needed
- Supports both training and inference-only modes via `IsTrainable` template parameter
- Same user-facing API as `MultilayerPerceptron`: `feedForward`, `trainNetwork`, `calculateError`, `getLearnedValues`

### Fixed-Point Arithmetic

- `QValue<IntegerBits, FractionalBits, IsSigned>` template supporting Q8.8, Q16.16, Q24.8, Q32.32, and other formats up to 128-bit
- Full operator overloading (`+`, `-`, `*`, `/`, comparisons)
- Configurable rounding (`TruncatePolicy`, `RoundUpPolicy`) and saturation (`WrapPolicy`, `MinMaxSaturatePolicy`) policies
- Pre-computed lookup tables for sigmoid, tanh, exp, and log across all supported bit-widths
- Also supports `float` and `double` as value types for prototyping

### Activation Functions

| Function | Policy Class | Range |
|----------|-------------|-------|
| Linear | `LinearActivationPolicy` | (-inf, inf) |
| ReLU | `ReluActivationPolicy` | [0, inf) |
| Capped ReLU | `CappedReluActivationPolicy` | [0, max] |
| Sigmoid | `SigmoidActivationPolicy` | (0, 1) |
| Tanh | `TanhActivationPolicy` | (-1, 1) |
| ELU | `EluActivationPolicy` | (-1, inf) |
| GELU | `GeluActivationPolicy` | (-0.17, inf) |
| SiLU | `SiLUActivationPolicy` | (-0.28, inf) |
| Softmax | `SoftmaxActivationPolicy` | (0, 1) per class |

Fixed-point activations use pre-computed lookup tables for speed. Floating-point activations use standard math functions.

### Q-Learning

- Tabular Q-learning with configurable reward and learning policies
- Deep Q-Network (DQN) with neural network function approximation
- Experience replay buffer for DQN training
- Dyna-Q hybrid model-free/model-based learning

## Quick Start

### Feed-Forward Network (XOR)

```cpp
#include "neuralnet.hpp"
#include "activationFunctions.hpp"

// Define a Q8.8 fixed-point XOR network
typedef tinymind::QValue<8, 8, true> ValueType;
typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>> TransferFunctions;

// 2 inputs, 3 hidden neurons, 1 output
typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<3>, 1,
    TransferFunctions> XorNetwork;

XorNetwork nn;
ValueType inputs[2], target[1];

// Training loop
for (int epoch = 0; epoch < 10000; ++epoch) {
    inputs[0] = 1; inputs[1] = 0; target[0] = 1;
    nn.feedForward(inputs);
    nn.trainNetwork(target);
}

// Inference
nn.feedForward(inputs);
ValueType output[1];
nn.getLearnedValues(output);
```

### Kolmogorov-Arnold Network (XOR)

```cpp
#include "kan.hpp"

// Define a Q8.8 fixed-point KAN
typedef tinymind::QValue<8, 8, true> ValueType;
typedef tinymind::KanTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    1> TransferFunctions;  // 1 output neuron

// 2 inputs, 5 hidden neurons, 1 output
// GridSize=5, SplineDegree=1 (piecewise linear, best for fixed-point)
typedef tinymind::KolmogorovArnoldNetwork<
    ValueType, 2, 1, 5, 1,
    TransferFunctions,
    true,  // trainable
    1,     // batch size
    5,     // grid size
    1      // spline degree
> KanNetwork;

KanNetwork kan;
ValueType inputs[2], target[1], output[1];

// Training loop (same API as MultilayerPerceptron)
for (int epoch = 0; epoch < 10000; ++epoch) {
    inputs[0] = 1; inputs[1] = 0; target[0] = 1;
    kan.feedForward(inputs);
    kan.trainNetwork(target);
}

// Inference
kan.feedForward(inputs);
kan.getLearnedValues(output);
```

### LSTM Network (Sequence Prediction)

```cpp
#include "neuralnet.hpp"

// Floating-point LSTM with 16 hidden neurons
typedef tinymind::LstmNeuralNetwork<double, 1,
    tinymind::HiddenLayers<16>, 1,
    FloatTransferFunctions> LstmNetwork;

LstmNetwork nn;
double input[1], target[1], output[1];

// Sequential training: feed one timestep at a time
for (int epoch = 0; epoch < 100000; ++epoch) {
    nn.resetState();  // clean state each epoch
    for (int t = 0; t < sequenceLength - 1; ++t) {
        input[0] = sequence[t];
        target[0] = sequence[t + 1];
        nn.feedForward(input);
        nn.trainNetwork(target);
    }
}
```

### Multi-Layer LSTM

```cpp
// Two LSTM hidden layers: 16 neurons -> 8 neurons -> output
typedef tinymind::LstmNeuralNetwork<double, 2,
    tinymind::HiddenLayers<16, 8>, 1,
    FloatTransferFunctions> DeepLstmNetwork;

// Three LSTM hidden layers
typedef tinymind::LstmNeuralNetwork<double, 2,
    tinymind::HiddenLayers<32, 16, 8>, 1,
    FloatTransferFunctions> DeeperLstmNetwork;
```

### GRU Network

```cpp
#include "neuralnet.hpp"

// GRU with 8 hidden neurons -- ~25% smaller than equivalent LSTM
typedef tinymind::GruNeuralNetwork<double, 2,
    tinymind::HiddenLayers<8>, 1,
    FloatTransferFunctions> GruNetwork;

GruNetwork nn;
double input[2], target[1], output[1];

// Sequential training (same API as LSTM)
for (int epoch = 0; epoch < 10000; ++epoch) {
    nn.resetState();
    for (int t = 0; t < sequenceLength - 1; ++t) {
        input[0] = sequence[t];
        input[1] = sequence[t + 1];
        target[0] = sequence[t + 2];
        nn.feedForward(input);
        nn.trainNetwork(target);
    }
}
```

### Training with Gradient Clipping and Weight Decay

```cpp
#include "fixedPointTransferFunctions.hpp"

typedef tinymind::QValue<8, 8, true> ValueType;

// Enable gradient clipping [-1, 1] and L2 weight decay (lambda ~ 1/256)
typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    1,                                                  // NumberOfOutputNeurons
    tinymind::DefaultNetworkInitializer<ValueType>,     // initializer
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>, // error calculator
    tinymind::ZeroToleranceCalculator<ValueType>,       // zero tolerance
    tinymind::GradientClipByValue<ValueType>,           // gradient clipping
    tinymind::L2WeightDecay<ValueType>,                 // weight decay
    tinymind::StepDecaySchedule<ValueType, 5000>        // LR decay every 5000 steps
> TransferFunctions;

typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<5>, 1,
    TransferFunctions> RegularizedNetwork;
```

### 1D Convolution (Sensor Feature Extraction)

```cpp
#include "conv1d.hpp"

// 100-point sensor input, kernel=5, stride=2, 8 filters
typedef tinymind::Conv1D<double, 100, 5, 2, 8> ConvType;
// Output: 8 filters * 48 positions = 384 features

ConvType conv;
conv.initializeWeights<RandomNumberGenerator>();

double sensorData[100];
double features[ConvType::OutputSize];  // 384

conv.forward(sensorData, features);

// Feed features into a classifier
typedef tinymind::NeuralNetwork<double, 384, tinymind::HiddenLayers<32>, 4,
    TransferFunctions> Classifier;
Classifier nn;
nn.feedForward(features);
```

### Conv1D + Pool1D + Dropout Pipeline

```cpp
#include "conv1d.hpp"
#include "pool1d.hpp"
#include "dropout.hpp"

// Conv1D: 100-point input, kernel=5, stride=1, 4 filters -> 96 * 4 = 384 outputs
typedef tinymind::Conv1D<double, 100, 5, 1, 4> ConvType;

// MaxPool1D: pool size 2, stride 2, 4 channels -> 48 * 4 = 192 outputs
typedef tinymind::MaxPool1D<double, ConvType::OutputLength, 2, 2, 4> PoolType;

// Dropout: 50% dropout on the 192 pooled features
typedef tinymind::Dropout<double, PoolType::OutputSize, 50> DropoutType;

ConvType conv;
PoolType pool;
DropoutType dropout;

double sensorData[100];
double convOut[ConvType::OutputSize];   // 384
double poolOut[PoolType::OutputSize];   // 192
double dropOut[PoolType::OutputSize];   // 192

// Forward pipeline
conv.forward(sensorData, convOut);
pool.forward(convOut, poolOut);
dropout.forward(poolOut, dropOut);  // applies mask during training

// Switch to inference (no dropout)
dropout.setTraining(false);
dropout.forward(poolOut, dropOut);  // identity pass-through
```

### Linear Self-Attention (Sequence Processing)

```cpp
#include "selfattention1d.hpp"

// 32 time steps, 16-dim embedding, 8-dim projections
typedef tinymind::SelfAttention1D<double, 32, 16, 8> AttnType;

AttnType attn;

// Set projection weights (W_q, W_k, W_v)
for (size_t proj = 0; proj < 3; ++proj)
    for (size_t r = 0; r < 16; ++r)
        for (size_t c = 0; c < 8; ++c)
            attn.setProjectionWeight(proj, r, c, randomWeight());

double sequence[32 * 16];  // input: 32 time steps x 16 features
double attended[32 * 8];   // output: 32 time steps x 8 features

attn.forward(sequence, attended);

// Feed into a classifier
classifier.feedForward(attended);
```

### Conv1D + Self-Attention Pipeline

```cpp
#include "conv1d.hpp"
#include "selfattention1d.hpp"

// Conv1D extracts features, self-attention models dependencies
typedef tinymind::Conv1D<double, 128, 5, 2, 4> ConvType;
// Conv output: 4 filters * 62 positions = 248 values
// Reshape as: 62 time steps x 4 features
typedef tinymind::SelfAttention1D<double, 62, 4, 4> AttnType;

ConvType conv;
AttnType attn;

double sensorData[128];
double convOut[ConvType::OutputSize];   // 248
double attnInput[62 * 4];              // reshaped
double attnOut[AttnType::OutputSize];  // 248

conv.forward(sensorData, convOut);
// Reshape filter-major to time-step-major...
attn.forward(attnInput, attnOut);
```

### Self-Attention with Fixed-Point (Q16.16)

```cpp
#include "selfattention1d.hpp"

typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
typedef tinymind::SelfAttention1D<ValueType, 16, 8, 4> AttnType;

AttnType attn;

// Set identity projections
for (size_t proj = 0; proj < 3; ++proj)
{
    for (size_t r = 0; r < 8; ++r)
        for (size_t c = 0; c < 4; ++c)
            attn.setProjectionWeight(proj, r, c,
                (r == c) ? ValueType(1, 0) : ValueType(0));
}

ValueType input[16 * 8];
ValueType output[16 * 4];
attn.forward(input, output);
```

### FFT (Frequency-Domain Feature Extraction)

```cpp
#include "fft1d.hpp"
#include <cmath>

// 128-point FFT for sensor signal analysis
const size_t N = 128;
tinymind::FFT1D<double, N> fft;

// Compute twiddle factors
double cosTable[N / 2], sinTable[N / 2];
for (size_t k = 0; k < N / 2; ++k)
{
    cosTable[k] = std::cos(-2.0 * M_PI * k / N);
    sinTable[k] = std::sin(-2.0 * M_PI * k / N);
}
fft.setTwiddleFactors(cosTable, sinTable);

double real[N] = { /* sensor samples */ };
double imag[N] = {}; // zero for real-valued input

fft.forward(real, imag);

// Compute power spectrum (avoids sqrt for fixed-point compatibility)
double magSq[N];
tinymind::FFT1D<double, N>::magnitudeSquared(real, imag, magSq);

// Feed N/2 frequency bins into a classifier
classifier.feedForward(magSq);
```

### Binary Dense Layer (Multiplication-Free)

```cpp
#include "binarylayer.hpp"

// 64 inputs, 16 outputs -- weights stored as packed bits (128 bytes vs 8 KB full-precision)
tinymind::BinaryDense<double, 64, 16> layer;

// Initialize latent weights (real-valued, used during training)
layer.setLatentWeight(0, 0, 0.5);   // will binarize to +1
layer.setLatentWeight(0, 1, -0.3);  // will binarize to -1
// ... set all latent weights

layer.binarizeWeights();  // pack sign(latent_weight) into bits

double input[64], output[16];
layer.forward(input, output);  // XNOR + popcount, no multiplication

// Training with straight-through estimator (STE)
double outputDeltas[16], inputDeltas[64];
layer.backward(outputDeltas, input, inputDeltas);
layer.updateWeights(-0.01);  // updates latent weights, then re-binarizes
```

### Ternary Dense Layer (Multiply-Free with Sparsity)

```cpp
#include "ternarylayer.hpp"

// 64 inputs, 16 outputs, 50% threshold -- weights are {-1, 0, +1}
tinymind::TernaryDense<double, 64, 16, 50> layer;

// Initialize latent weights
layer.setLatentWeight(0, 0, 0.9);   // large magnitude -> +1 or -1
layer.setLatentWeight(0, 1, 0.01);  // small magnitude -> 0 (pruned)
// ... set all latent weights

layer.ternarizeWeights();  // threshold-based: |w| < thresh*mean -> 0

double input[64], output[16];
layer.forward(input, output);  // conditional add/subtract/skip, no multiply

// Training with STE
double outputDeltas[16], inputDeltas[64];
layer.backward(outputDeltas, input, inputDeltas);
layer.updateWeights(-0.01);
```

### RMSprop Optimizer

```cpp
#include "rmsprop.hpp"

// Floating-point RMSprop
typedef FloatingPointTransferFunctions<
    double, RandomNumberGenerator,
    tinymind::TanhActivationPolicy,
    tinymind::TanhActivationPolicy> BaseTF;

struct RmsPropTF : public BaseTF
{
    typedef tinymind::RmsPropOptimizerFloat<double> OptimizerPolicyType;
};

typedef tinymind::MultilayerPerceptron<double, 2, 1, 5, 1, RmsPropTF> Network;

// Fixed-point RMSprop (Q8.8: decay ≈ 230/256 ≈ 0.898, epsilon ≈ 1/256)
typedef tinymind::QValue<8, 8, true> QType;
typedef tinymind::FixedPointTransferFunctions<
    QType, RandomNumberGenerator<QType>,
    tinymind::TanhActivationPolicy<QType>,
    tinymind::TanhActivationPolicy<QType>,
    1,                                                  // NumberOfOutputNeurons
    tinymind::DefaultNetworkInitializer<QType>,
    tinymind::MeanSquaredErrorCalculator<QType, 1>,
    tinymind::ZeroToleranceCalculator<QType>,
    tinymind::NullGradientClippingPolicy<QType>,
    tinymind::NullWeightDecayPolicy<QType>,
    tinymind::FixedLearningRatePolicy<QType>,
    tinymind::RmsPropOptimizer<QType>                   // RMSprop optimizer
> FixedPointTF;
```

### Truncated BPTT (Recurrent Training)

```cpp
#include "truncatedBPTT.hpp"

// LSTM with truncated BPTT over 5-step windows
typedef tinymind::LstmNeuralNetwork<double, 1,
    tinymind::HiddenLayers<16>, 1,
    FloatTransferFunctions> LstmNetwork;

LstmNetwork nn;
tinymind::TruncatedBPTT<LstmNetwork, 5> trainer;

for (int epoch = 0; epoch < 10000; ++epoch) {
    nn.resetState();
    trainer.reset();
    for (int t = 0; t < sequenceLength - 1; ++t) {
        input[0] = sequence[t];
        target[0] = sequence[t + 1];
        trainer.step(nn, input, target);  // trains every 5 steps
    }
    trainer.flush(nn);  // train on remaining steps
}
```

### Q-Learning (Maze)

```cpp
#include "qlearn.hpp"

typedef tinymind::QLearningEnvironment<
    uint8_t,    // state type
    uint8_t,    // action type
    double,     // value type
    6,          // number of states
    6,          // number of actions
    RandomPolicy,
    LearningPolicy> MazeEnvironment;

MazeEnvironment env;
// Run episodes, update Q-values...
```

## Network Types

| Type | Class | Description |
|------|-------|-------------|
| Feed-forward | `NeuralNetwork` | Standard MLP with configurable layers (`MultilayerPerceptron` alias for uniform layers) |
| 1D Convolution | `Conv1D` | Time-series feature extraction with configurable kernel/stride/filters |
| Max Pooling | `MaxPool1D` | Downsampling via maximum value selection with argmax tracking |
| Average Pooling | `AvgPool1D` | Downsampling via mean with uniform gradient distribution |
| Dropout | `Dropout` | Inverted dropout regularization with training/inference mode |
| Self-Attention | `SelfAttention1D` | Linear attention with ReLU kernel feature map |
| FFT | `FFT1D` | Radix-2 FFT with scaled butterfly for frequency-domain feature extraction |
| Binary Dense | `BinaryDense` | XNOR+popcount dense layer with 1-bit packed weights |
| Ternary Dense | `TernaryDense` | Multiply-free dense layer with 2-bit packed {-1,0,+1} weights |
| KAN | `KolmogorovArnoldNetwork` | Learnable B-spline activations on edges |
| Elman RNN | `ElmanNeuralNetwork` | Simple recurrent with depth-1 feedback |
| Recurrent | `RecurrentNeuralNetwork` | Configurable recurrent connection depth |
| LSTM | `LstmNeuralNetwork` | Long Short-Term Memory with 4 gates |
| GRU | `GruNeuralNetwork` | Gated Recurrent Unit with 3 gates |

All network types support both fixed-point and floating-point value types.

## Architecture

### Policy-Based Design

Every aspect of the network is controlled by template parameters:

```
NeuralNetwork<
    ValueType,              // QValue<8,8,true>, double, float
    NumberOfInputs,         // compile-time input count
    HiddenLayersDescriptor, // HiddenLayers<N0, N1, ...>
    NumberOfOutputs,        // compile-time output count
    TransferFunctionsPolicy,// activation + training policy
    IsTrainable,            // true/false (inference-only mode)
    BatchSize,              // gradient accumulation batch size
    HasRecurrentLayer,      // enables recurrent connections
    HiddenLayerConfig,      // NonRecurrent/Recurrent/GRU/LSTM
    RecurrentConnectionDepth,
    OutputLayerConfiguration // FeedForward/Classifier(softmax)
>
```

### Zero Overhead

The heterogeneous layer chain (`LayerChain`/`EmptyLayerChain`) compiles to the exact same binary size as uniform array-based storage:

**MLP sizes (double):**

| Configuration | Size (bytes) |
|---|---|
| 2 -> 5 -> 1 (1 hidden) | 1,008 |
| 2 -> 3 -> 1 (Elman RNN) | 1,056 |
| 2 -> 5 -> 1 (non-trainable) | 360 |

**KAN vs MLP XOR comparison (Q8.8):**

| | MLP [2]->[3]->[1] | KAN [2]->[5]->[1] G=5 k=1 |
|---|---|---|
| Trainable | 328 bytes | 1,192 bytes |
| Inference-only | 144 bytes | 416 bytes |
| Trainable params | 13 weights | 120 (coefficients + edge weights) |
| Params per edge | 1 scalar | 8 (6 spline coefficients + w_b + w_s) |

## Building

### Requirements

- C++14 or later
- [Boost.Test](https://www.boost.org/doc/libs/release/libs/test/) (for unit tests only)
- Set `BOOST_HOME` environment variable to your Boost installation path

### Build and Run Tests

```bash
# Build and run everything
make check

# Individual test suites
cd unit_test/nn && make clean && make && make run
cd unit_test/qformat && make clean && make && make run
cd unit_test/qlearn && make clean && make && make run
```

### Build Examples

```bash
cd examples/xor && make clean && make
cd examples/kan_xor && make clean && make
cd examples/gru_xor && make clean && make
cd examples/lstm_sinusoid && make clean && make
cd examples/maze && make clean && make
cd examples/dqn_maze && make clean && make
```

### Compiler Flags

- Debug: `-Wall -Wextra -Werror -Wpedantic -ggdb`
- Release: `-Wall -Wextra -Werror -Wpedantic -O3`

## Project Structure

```
tinymind/
  cpp/                          # Core library headers
    neuralnet.hpp               # Neural network templates (~5700 lines)
    kan.hpp                     # Kolmogorov-Arnold Network templates
    bspline.hpp                 # B-spline evaluation engine (De Boor algorithm)
    kanTransferFunctions.hpp    # KAN transfer functions and SiLU activation
    conv1d.hpp                  # 1D convolution layer
    selfattention1d.hpp         # Linear self-attention layer
    fft1d.hpp                   # Radix-2 FFT with compile-time bit-reversal tables
    binarylayer.hpp             # Binary neural network layer (XNOR+popcount)
    ternarylayer.hpp            # Ternary neural network layer ({-1,0,+1} weights)
    qformat.hpp                 # Fixed-point arithmetic
    qlearn.hpp                  # Q-learning and DQN
    activationFunctions.hpp     # Activation function policies (9 functions)
    fixedPointTransferFunctions.hpp
    adam.hpp                    # Adam optimizer policy
    rmsprop.hpp                 # RMSprop optimizer policy
    pool1d.hpp                  # MaxPool1D and AvgPool1D layers
    dropout.hpp                 # Inverted dropout regularization layer
    gradientClipping.hpp        # Gradient clipping policies
    weightDecay.hpp             # L2 weight decay policies
    learningRateSchedule.hpp    # Learning rate scheduling policies
    earlyStopping.hpp           # Early stopping convergence monitor
    teacherForcing.hpp          # Scheduled sampling for recurrent training
    truncatedBPTT.hpp           # Truncated BPTT training utility
    networkStats.hpp            # Compile-time network statistics
    xavier.hpp                  # Xavier weight initialization
    lookupTables.cpp            # Pre-computed activation tables (~3MB)
    include/                    # Support headers
      nnproperties.hpp          # Weight file manager (MLP, LSTM, GRU, KAN)
      constants.hpp, limits.hpp, random.hpp, ...
  examples/
    xor/                        # MLP XOR gate learning
    kan_xor/                    # KAN XOR gate learning
    gru_xor/                    # GRU XOR gate learning
    lstm_sinusoid/              # LSTM sinusoid prediction
    maze/                       # Tabular Q-learning maze solver
    dqn_maze/                   # Deep Q-Network maze solver
    pytorch/                    # PyTorch weight import (MLP + GRU export)
  unit_test/
    nn/                         # Neural network tests (147 test cases)
    kan/                        # KAN tests (16 test cases)
    qformat/                    # Fixed-point type tests (static_assert)
    qlearn/                     # Q-learning tests
  apps/
    activation/                 # Lookup table generator tool
```

## Documentation

- [CLAUDE.md](CLAUDE.md) -- Architecture overview and build commands
- [KAN.md](KAN.md) -- KAN implementation plan and summary
- [LSTM.md](LSTM.md) -- LSTM implementation analysis and improvement roadmap

## License

MIT License. See individual source files for copyright notices.
