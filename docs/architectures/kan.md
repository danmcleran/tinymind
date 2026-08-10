---
title: Kolmogorov-Arnold Networks
layout: default
parent: Architectures
nav_order: 2
---

# Kolmogorov-Arnold Networks (KAN)

> **Real-world use:** an on-device soft sensor replaces a hand-tuned calibration lookup table. A small KAN learns the smooth, nonlinear response curve that maps a raw gas or temperature reading to a corrected value, fitting the curve in its B-spline edges in ~1.2 KB — adapting to the part instead of shipping a fixed table.

Tinymind implements [Kolmogorov-Arnold Networks](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem) (KAN), a neural network architecture where learnable activation functions are placed on the edges (connections) rather than at the nodes. In a KAN, each edge has its own B-spline activation function, and nodes simply sum their inputs. This is in contrast to standard MLPs where edges carry scalar weights and nodes apply a shared activation function.

KAN can be more parameter-efficient than MLP for certain smooth, low-dimensional functions, learning the activation shape rather than relying on fixed activations with learned weights.

![KAN XOR learning curve]({{ site.baseurl }}/assets/plots/kan_xor_learning_curve.png)

*`examples/kan_xor` (`make run && make plot`): the KAN learns XOR by fitting its B-spline edge activations.*

## KAN on Embedded: Trading Memory for Accuracy

KAN uses more memory per connection than MLP (8 parameters per edge vs 1 weight at `GridSize=5, SplineDegree=1`), so at the *same* topology it is roughly 2.7x larger trainable and 2.1x larger inference-only. It is not 8x, because both networks carry the same neuron and layer scaffolding. KAN can sometimes approximate smooth functions with fewer neurons than an equivalent MLP, offsetting part of the per-edge cost.

A trainable KAN (2->5->1) in Q8.8 fixed-point is 1,200 bytes -- well within reach for any ARM Cortex-M class device. For inference-only deployment (`IsTrainable=false`), this drops to 416 bytes.

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
// Q16.16: KAN training needs more range and precision than Q8.8 provides.
// With 8 learnable parameters per edge accumulating gradients, Q8.8 saturates
// and fails to converge. Inference has no gradients to accumulate, so a
// deployed (IsTrainable=false) network can drop back to Q8.8.
typedef tinymind::QValue<16, 16, true> ValueType;

static const size_t GRID_SIZE = 5;
static const size_t SPLINE_DEGREE = 1; // piecewise linear -- best for fixed-point

typedef tinymind::KanTransferFunctions<ValueType,
                                       RandomNumberGenerator,
                                       1,
                                       KanNetworkInitializer> TransferFunctionsType;

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

`KanNetworkInitializer` is a custom network initializer supplying a lower learning rate (~0.03125), momentum (~0.0625), and acceleration than the MLP defaults. A KAN edge has 8 learnable parameters instead of 1, and the default rates drive it to diverge.

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
make          # debug build
make release  # optimized build
make run      # runs the corner-trained and --dense variants, writes CSVs
make plot     # renders the learning curve
```

`make run` executes both modes. The default trains on the four crisp XOR corners; `--dense` samples the whole `[0,1]^2` input region and labels by the XOR of the two halves, which constrains the splines everywhere and yields a smooth decision surface. Average error falls from ~0.38 at iteration 100 to ~0.004 at iteration 20,000.

# Size Comparison: KAN vs MLP

Matched topology, `sizeof` of the whole network object. The KAN is `GridSize=5, SplineDegree=1`.

| | MLP [2]->[5]->[1] | KAN [2]->[5]->[1] G=5 k=1 |
|---|---|---|
| Trainable (Q8.8) | 440 bytes | 1,200 bytes |
| Non-trainable (Q8.8) | 200 bytes | 416 bytes |
| Trainable (Q16.16) | 632 bytes | 2,200 bytes |
| Non-trainable (Q16.16) | 224 bytes | 696 bytes |
| Trainable (double) | 1,008 bytes | 4,208 bytes |
| Non-trainable (double) | 360 bytes | 1,256 bytes |
| Parameters per edge | 1 scalar weight | 8 (6 coefficients + w_b + w_s) |

Trainable storage is roughly 3x inference storage: `TrainableKanConnection` adds a gradient, a delta weight, and a previous delta weight for *every* learnable parameter. Setting `IsTrainable=false` drops all of it.

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
