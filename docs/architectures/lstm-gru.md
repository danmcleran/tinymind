---
title: LSTM & GRU
layout: default
parent: Architectures
nav_order: 1
---

# LSTM and GRU Recurrent Networks

Tinymind provides three recurrent neural network architectures for learning from sequential data: Elman (simple recurrent), LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit). All are implemented as C++ templates and support both fixed-point and floating-point value types.

Recurrent networks maintain internal state across time steps, making them suitable for tasks like sequence prediction, time-series forecasting, and temporal pattern recognition. The key architectural difference from feed-forward networks is that hidden neurons receive feedback connections from the previous time step.

## Embedded Use Cases

On resource-constrained embedded systems, recurrent networks enable on-device temporal intelligence without cloud connectivity:

- **Wearable health monitoring** -- ECG arrhythmia detection, heart rate prediction, sleep stage classification running continuously on a battery-powered sensor
- **Predictive maintenance** -- vibration pattern analysis on industrial equipment, detecting bearing wear or motor degradation before failure
- **Sensor time-series** -- temperature/pressure trend prediction on IoT nodes, enabling local decision-making without network round-trips
- **Embedded control** -- adaptive motor control, robotic joint coordination, and real-time signal processing

A trainable GRU (2->3->1) in Q8.8 fixed-point takes just 808 bytes -- small enough to run on virtually any microcontroller, with no FPU, GPU, or OS required. For inference-only deployment after training in PyTorch, the memory footprint drops further to ~336 bytes.

# Recurrent Network Templates

## ElmanNeuralNetwork

The simplest recurrent architecture. A single hidden layer receives feedback from its own output at the previous time step. Recurrent connection depth is fixed to 1.

```cpp
template<
    typename ValueType,
    size_t NumberOfInputs,
    size_t NumberOfNeuronsInHiddenLayer,
    size_t NumberOfOutputs,
    typename TransferFunctionsPolicy,
    bool IsTrainable = true,
    size_t BatchSize = 1,
    outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
>
class ElmanNeuralNetwork
```

## LstmNeuralNetwork

LSTM networks use 4 gates (input, forget, output, cell candidate) to control information flow. This allows them to learn long-term dependencies that simple recurrent networks struggle with. LSTM supports multi-layer configurations via `HiddenLayers<N0, N1, ...>`.

```cpp
template<
    typename ValueType,
    size_t NumberOfInputs,
    typename HiddenLayersDescriptor,
    size_t NumberOfOutputs,
    typename TransferFunctionsPolicy,
    bool IsTrainable = true,
    size_t BatchSize = 1,
    size_t RecurrentConnectionDepth = 1,
    outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
>
class LstmNeuralNetwork
```

## GruNeuralNetwork

GRU networks use 3 gates (update, reset, candidate) -- simpler than LSTM's 4 gates. GRU uses ~25% less memory per hidden neuron than LSTM while achieving comparable performance on many tasks. GRU is often preferred for resource-constrained embedded systems.

```cpp
template<
    typename ValueType,
    size_t NumberOfInputs,
    typename HiddenLayersDescriptor,
    size_t NumberOfOutputs,
    typename TransferFunctionsPolicy,
    bool IsTrainable = true,
    size_t BatchSize = 1,
    size_t RecurrentConnectionDepth = 1,
    outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
>
class GruNeuralNetwork
```

## Template Parameters

**ValueType** - The numeric type used by the network. Can be a `QValue` fixed-point type, `float`, or `double`.

**NumberOfInputs** - Number of input neurons.

**HiddenLayersDescriptor** - Specifies hidden layer sizes. Use `HiddenLayers<N>` for a single hidden layer with N neurons, or `HiddenLayers<N0, N1, ...>` for multiple hidden layers with different sizes.

**NumberOfOutputs** - Number of output neurons.

**TransferFunctionsPolicy** - Policy class providing activation functions, random number generation, optimizer, error calculation, gradient clipping, weight decay, and learning rate schedule.

**IsTrainable** - When `false`, training code is omitted entirely, reducing binary size. Non-trainable networks can still load pre-trained weights.

**BatchSize** - Number of samples to accumulate before back-propagation.

**RecurrentConnectionDepth** - Number of previous time steps stored in recurrent connections.

**OutputLayerConfiguration** - `FeedForwardOutputLayerConfiguration` for regression, `ClassifierOutputLayerConfiguration` for softmax classification.

# Hidden Layer Configuration

Tinymind supports heterogeneous hidden layer sizes using the `HiddenLayers` variadic template:

```cpp
// Single hidden layer with 16 neurons
typedef tinymind::LstmNeuralNetwork<ValueType, 1,
    tinymind::HiddenLayers<16>, 1,
    TransferFunctionsType> SingleLayerLstm;

// Two hidden layers: 16 neurons then 8 neurons
typedef tinymind::LstmNeuralNetwork<ValueType, 2,
    tinymind::HiddenLayers<16, 8>, 1,
    TransferFunctionsType> TwoLayerLstm;

// Three hidden layers: 32 -> 16 -> 8
typedef tinymind::LstmNeuralNetwork<ValueType, 2,
    tinymind::HiddenLayers<32, 16, 8>, 1,
    TransferFunctionsType> ThreeLayerLstm;
```

# LSTM Example: Sinusoid Prediction

This example trains an LSTM to predict the next value in a sinusoidal sequence. Source code: [lstm_sinusoid.cpp](https://github.com/danmcleran/tinymind/blob/master/examples/lstm_sinusoid/lstm_sinusoid.cpp).

## Network Definition

```cpp
typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomGen<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>,
    1,
    tinymind::DefaultNetworkInitializer<ValueType>,
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType>> TransferFunctionsType;

typedef tinymind::LstmNeuralNetwork<
    ValueType, 1,
    tinymind::HiddenLayers<16>,
    1,
    TransferFunctionsType> LstmNetworkType;
```

## Training Loop

```cpp
LstmNetworkType lstmNet;
ValueType input[1], target[1], error;

for (unsigned epoch = 0; epoch < TRAINING_EPOCHS; ++epoch)
{
    for (size_t t = 0; t < NUM_SAMPLES - 1; ++t)
    {
        input[0] = sinSamples[t];
        target[0] = sinSamples[t + 1];

        lstmNet.feedForward(&input[0]);
        error = lstmNet.calculateError(&target[0]);

        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            lstmNet.trainNetwork(&target[0]);
        }
    }
}
```

## Building The Example

```bash
cd examples/lstm_sinusoid
make        # debug build
make release # optimized build
cd output
./lstm_sinusoid
```

# GRU Example: XOR

This example trains a GRU to predict the XOR function with early stopping. Source code: [gru_xor.cpp](https://github.com/danmcleran/tinymind/blob/master/examples/gru_xor/gru_xor.cpp).

## Training With Early Stopping

```cpp
GruNetworkType gruNet;
tinymind::EarlyStopping<ValueType, 5000> stopper;

for (unsigned i = 0; i < TRAINING_ITERATIONS; ++i)
{
    generateXorValues(values[0], values[1], output[0]);

    gruNet.feedForward(&values[0]);
    error = gruNet.calculateError(&output[0]);

    if (!TransferFunctionsType::isWithinZeroTolerance(error))
    {
        gruNet.trainNetwork(&output[0]);
    }

    if (stopper.shouldStop(error))
    {
        break; // converged, stop training early
    }
}
```

# Size Comparison

| Architecture | Hidden | Trainable (double) | Trainable (Q8.8) | Non-trainable (Q8.8) |
|---|---|---|---|---|
| MLP (2->5->1) | 5 | 1,008 bytes | 328 bytes | 144 bytes |
| Elman (2->3->1) | 3 | 1,056 bytes | 472 bytes | 192 bytes |
| LSTM (2->3->1) | 3 | 3,024 bytes | 952 bytes | 384 bytes |
| GRU (2->3->1) | 3 | 2,400 bytes | 808 bytes | 336 bytes |

GRU uses ~25% less memory than LSTM (3 gates vs 4 gates). Even a trainable LSTM in Q8.8 fixed-point fits in under 1 KB.

# Resetting State

Recurrent networks accumulate internal state across time steps. When starting a new sequence, reset the state:

```cpp
lstmNet.resetState();  // clears cell state and hidden state
gruNet.resetState();   // clears hidden state
```

# Weight Import/Export

Trained recurrent network weights can be saved and loaded using `RecurrentNetworkPropertiesFileManager`:

```cpp
typedef tinymind::RecurrentNetworkPropertiesFileManager<LstmNetworkType> FileManager;

// Save weights
std::ofstream outFile("lstm_weights.txt");
FileManager::storeNetworkWeights(lstmNet, outFile);

// Load weights
std::ifstream inFile("lstm_weights.txt");
FileManager::template loadNetworkWeights<ValueType, ValueType>(lstmNet, inFile);
```

See the [Weight Import Export and PyTorch Interoperability]({{ site.baseurl }}/training/pytorch-interop) page for details on the weight file format and PyTorch export scripts.
