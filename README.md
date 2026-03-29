# TinyMind

A header-only C++ template library for neural networks, Kolmogorov-Arnold Networks (KAN), LSTM and GRU recurrent networks, and Q-learning, designed for embedded systems with no FPU, GPU, or vectorized instruction requirements.

Inspired by Andrei Alexandrescu's policy-based design from [Modern C++ Design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design), TinyMind uses template metaprogramming to produce zero-overhead abstractions where network topology, value type, activation functions, and training policies are all compile-time parameters.

## Features

### Neural Networks

- **Feed-forward networks** with arbitrary depth and width
- **Kolmogorov-Arnold Networks (KAN)** with learnable B-spline activation functions on edges
- **Recurrent neural networks** (Elman) with configurable recurrent connection depth
- **LSTM networks** with gated cell state, supporting single and multi-layer configurations
- **GRU networks** (Gated Recurrent Unit) with 3-gate architecture -- ~25% less memory than LSTM per hidden neuron
- **Heterogeneous hidden layers** via `HiddenLayers<N0, N1, ...>` for different neuron counts per layer
- **Batch training** with configurable batch size
- **Softmax output** for multi-class classification
- **Xavier weight initialization** (uniform and normal distributions)
- **Weight import/export** in CSV and binary formats (interoperable with PyTorch)

### Training Policies

- **Gradient clipping** via configurable `GradientClipByValue` policy to prevent exploding gradients (especially critical for fixed-point)
- **L2 weight decay** (ridge regularization) via `L2WeightDecay` policy to prevent weight overflow and reduce overfitting
- **Learning rate scheduling** with `StepDecaySchedule` (multiply by decay factor every N steps) and `FixedLearningRatePolicy` (default)
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
| Feed-forward | `NeuralNetwork` | Standard MLP with configurable layers |
| Feed-forward (uniform) | `MultilayerPerceptron` | MLP with equal-sized hidden layers |
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

**MLP sizes (Q8.8):**

| Configuration | Size (bytes) |
|---|---|
| 2 -> 5 -> 1 (1 hidden) | 1,000 |
| 2 -> 5 -> 5 -> 1 (2 hidden) | 2,104 |
| 10 -> 20 -> 20 -> 5 (large) | 25,480 |
| 2 -> 3 -> 1 (Elman RNN) | 1,048 |
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
    neuralnet.hpp               # Neural network templates (~5600 lines)
    kan.hpp                     # Kolmogorov-Arnold Network templates
    bspline.hpp                 # B-spline evaluation engine (De Boor algorithm)
    kanTransferFunctions.hpp    # KAN transfer functions and SiLU activation
    qformat.hpp                 # Fixed-point arithmetic
    qlearn.hpp                  # Q-learning and DQN
    activationFunctions.hpp     # Activation function policies
    fixedPointTransferFunctions.hpp
    gradientClipping.hpp        # Gradient clipping policies
    weightDecay.hpp             # L2 weight decay policies
    learningRateSchedule.hpp    # Learning rate scheduling policies
    xavier.hpp                  # Xavier weight initialization
    lookupTables.cpp            # Pre-computed activation tables (~3MB)
    include/                    # Support headers
      nnproperties.hpp          # Network property/weight file manager
      constants.hpp, limits.hpp, random.hpp, ...
  examples/
    xor/                        # MLP XOR gate learning
    kan_xor/                    # KAN XOR gate learning
    maze/                       # Tabular Q-learning maze solver
    dqn_maze/                   # Deep Q-Network maze solver
    pytorch/                    # PyTorch weight import for inference
  unit_test/
    nn/                         # Neural network tests (71 test cases)
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
