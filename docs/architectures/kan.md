---
title: Kolmogorov-Arnold Networks
layout: default
parent: Architectures
nav_order: 2
---

# Kolmogorov-Arnold Networks (KAN)

Tinymind implements [Kolmogorov-Arnold Networks](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem) (KAN), a neural network architecture where learnable activation functions are placed on the edges (connections) rather than at the nodes. In a KAN, each edge has its own B-spline activation function, and nodes simply sum their inputs. This is in contrast to standard MLPs where edges carry scalar weights and nodes apply a shared activation function.

KAN can be more parameter-efficient than MLP for certain smooth, low-dimensional functions, learning the activation shape rather than relying on fixed activations with learned weights.

## KAN on Embedded: Trading Memory for Accuracy

KAN uses more memory per connection than MLP (8 parameters per edge vs 1 weight), so it is 3-4x larger for the same topology. However, KAN can sometimes approximate smooth functions with fewer neurons than an equivalent MLP, potentially offsetting the per-edge cost.

Even at its largest, a trainable KAN (2->5->1) in Q8.8 fixed-point is 1,192 bytes -- still well within reach for any ARM Cortex-M class device. For inference-only deployment (`IsTrainable=false`), this drops to 416 bytes.

For fixed-point targets, always use `SplineDegree=1` (piecewise linear). Higher-degree polynomials involve multi-step intermediate computations that risk overflow in Q-format arithmetic.

# Template Declaration

```cpp
template<
    typename ValueType,
    size_t NumberOfInputs,
    size_t NumberOfHiddenLayers,
    size_t NumberOfNeuronsInHiddenLayers,
    size_t NumberOfOutputs,
    typename TransferFunctionsPolicy,
    bool IsTrainable = true,
    size_t BatchSize = 1,
    size_t GridSize = 5,
    size_t SplineDegree = 3
>
class KolmogorovArnoldNetwork
```

**Template Parameters:**
- `ValueType` - Numeric type (`QValue`, `float`, or `double`)
- `NumberOfInputs` - Number of input neurons
- `NumberOfHiddenLayers` - Number of hidden layers (>= 1)
- `NumberOfNeuronsInHiddenLayers` - Neurons per hidden layer
- `NumberOfOutputs` - Number of output neurons
- `TransferFunctionsPolicy` - `KanTransferFunctions<...>` policy class
- `IsTrainable` - Enable/disable training (non-trainable mode saves memory)
- `BatchSize` - Gradient accumulation batch size
- `GridSize` - Number of B-spline grid intervals (default 5)
- `SplineDegree` - B-spline polynomial degree (default 3)

# Edge Function

Each KAN edge computes:

```
phi(x) = w_b * SiLU(x) + w_s * spline(x)
```

Where:
- `w_b` is the base weight (scalar)
- `SiLU(x) = x * sigmoid(x)` is the residual activation (reuses existing sigmoid lookup tables)
- `w_s` is the spline weight (scalar)
- `spline(x)` is a B-spline evaluated using `GridSize + SplineDegree` learnable coefficients

Each edge stores `2 + GridSize + SplineDegree` learnable parameters. For `GridSize=5, SplineDegree=1` (piecewise linear), that's 8 parameters per edge vs 1 weight per edge in an MLP.

# B-Spline Evaluation

Tinymind uses the De Boor algorithm for efficient B-spline evaluation. The `UniformKnotVector` template generates evenly spaced knot vectors at initialization.

```cpp
template<typename ValueType, size_t GridSize, size_t SplineDegree>
struct UniformKnotVector
{
    static const size_t NumberOfKnots = GridSize + 2 * SplineDegree + 1;
    static const size_t NumberOfBasisFunctions = GridSize + SplineDegree;
    // ...
};
```

# KAN Transfer Functions

KAN uses its own transfer functions policy class:

```cpp
template<
    typename ValueType,
    class KanRandomNumberGeneratorPolicy,
    unsigned NumberOfOutputNeurons = 1,
    class KanNetworkInitializationPolicy = tinymind::DefaultNetworkInitializer<ValueType>,
    class KanErrorCalculatorPolicy = tinymind::MeanSquaredErrorCalculator<ValueType, NumberOfOutputNeurons>,
    class KanZeroTolerancePolicy = tinymind::ZeroToleranceCalculator<ValueType>
>
struct KanTransferFunctions
```

# Example: KAN XOR

Source code: [examples/kan_xor/](https://github.com/danmcleran/tinymind/tree/master/examples/kan_xor)

## Network Definition

```cpp
typedef tinymind::QValue<8, 8, true> ValueType;

static const size_t GRID_SIZE = 5;
static const size_t SPLINE_DEGREE = 1; // piecewise linear -- best for fixed-point

typedef tinymind::KanTransferFunctions<ValueType,
                                       RandomNumberGenerator,
                                       1> TransferFunctionsType;

typedef tinymind::KolmogorovArnoldNetwork<ValueType,
                                          2,             // inputs
                                          1,             // hidden layers
                                          5,             // neurons per hidden layer
                                          1,             // outputs
                                          TransferFunctionsType,
                                          true,          // trainable
                                          1,             // batch size
                                          GRID_SIZE,
                                          SPLINE_DEGREE> KanNetworkType;
```

## Training Loop

The KAN API is identical to `MultilayerPerceptron`:

```cpp
KanNetworkType testKanNet;
ValueType values[2], output[1], learnedValues[1], error;

for (unsigned i = 0; i < 20000; ++i)
{
    generateXorTrainingValue(values[0], values[1], output[0]);

    testKanNet.feedForward(&values[0]);
    error = testKanNet.calculateError(&output[0]);

    if (!KanNetworkType::KanTransferFunctionsPolicy::isWithinZeroTolerance(error))
    {
        testKanNet.trainNetwork(&output[0]);
    }

    testKanNet.getLearnedValues(&learnedValues[0]);
}
```

## Building The Example

```bash
cd examples/kan_xor
make        # debug build
make release # optimized build
cd output
./kan_xor
```

# Size Comparison: KAN vs MLP

| | MLP [2]->[3]->[1] | KAN [2]->[5]->[1] G=5 k=1 |
|---|---|---|
| Trainable (Q8.8) | 328 bytes | 1,192 bytes |
| Non-trainable (Q8.8) | 144 bytes | 416 bytes |
| Trainable (double) | 1,008 bytes | 4,208 bytes |
| Parameters per edge | 1 scalar weight | 8 (6 coefficients + w_b + w_s) |

# Weight Import/Export

KAN weights can be saved and loaded using `KanNetworkPropertiesFileManager`:

```cpp
typedef tinymind::KanNetworkPropertiesFileManager<KanNetworkType> FileManager;

// Save
std::ofstream outFile("kan_weights.txt");
FileManager::storeNetworkWeights(testKanNet, outFile);

// Load
std::ifstream inFile("kan_weights.txt");
FileManager::template loadNetworkWeights<ValueType, ValueType>(testKanNet, inFile);
```

# When To Use KAN vs MLP

- **KAN** excels at learning smooth, low-dimensional functions with fewer neurons than an equivalent MLP. The learnable activation shape on each edge gives KAN more expressiveness per connection.
- **MLP** is more memory-efficient per connection (1 weight vs 8+ parameters per edge) and benefits from decades of optimization. For problems where fixed activations like tanh or ReLU are sufficient, MLP is the better choice on embedded systems.
- For fixed-point targets, always use `SplineDegree=1` (piecewise linear) to avoid overflow from higher-order polynomial terms.
