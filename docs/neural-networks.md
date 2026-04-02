---
title: Neural Networks
layout: default
nav_order: 2
---

# Neural Networks

Years ago, I began writing C++ template libraries with the goal of instantiating and running neural networks and machine learning algorithms within embedded systems. I did not have an FPU (floating point unit), GPU (graphics processing unit), or any vectorized instructions(e.g. SIMD) at my disposal so I needed to ensure I could run these algorithms with only a very simple CPU. Memory to store code and data was also highly constrained in my environment. I didn't have room to store code and data that I wasn't going to use.

My inspiration for these libraries was Andrei Alexandrescu's [Modern C++ Design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design). This book was my first exposure to the power of C++ templates and [template metaprogramming](https://en.wikipedia.org/wiki/Template_metaprogramming). In the book he describes how to design your code using policy classes as template parameters to customize behavior. This fit my needs perfectly since I wanted the code to be scalable from one problem to the next (e.g. resolution, activation function(s), etc.) I also used this technique to make the code within these libraries as small and efficient as possible.

# Overview

Neural networks within [tinymind](https://github.com/danmcleran/tinymind) are implemented using C++ templates and customized via template parameters, both types as well as policy classes. Since the code is implemented using C++ templates, this allows us to instantiate neural networks within a very small code and data footprint. You only compile what you need and do not have to lug around a huge runtime library which contains a bunch of code you may never use. You can define a very specific set of neural network(s) for a very specific set of type(s) and that's all you pay for.

## Compile-Time Specialization: You Only Pay for What You Use

Traditional ML frameworks (TensorFlow Lite, ONNX Runtime, etc.) ship a runtime interpreter that supports every possible layer type, activation function, and data type. You link this entire runtime even if you only use a fraction of it. On embedded systems with 16-64 KB of flash, this overhead alone can consume the entire budget.

Tinymind takes the opposite approach. Because everything is a C++ template, the compiler generates machine code only for the exact configuration you instantiate:

- **Using only tanh activation?** Only the tanh lookup table is compiled (192 bytes for Q8.8). Sigmoid, exp, and log tables are excluded entirely.
- **Inference only (no training)?** Set `IsTrainable=false` and all backpropagation code, gradient storage, and delta weight tracking are eliminated. A 1,008-byte trainable MLP becomes a 360-byte inference engine -- a 64% reduction.
- **Single hidden layer?** Multi-layer management code is never generated.
- **No recurrent connections?** LSTM/GRU gate logic is never compiled.

This is why a complete XOR neural network (code + data + lookup tables) fits in [under 4KB]({{ site.baseurl }}/getting-started/xor-under-4kb) and a Q-learner fits in [under 1KB]({{ site.baseurl }}/getting-started/q-learning-under-1kb). There is no minimum library overhead -- the floor is determined entirely by your network's topology and type.

## Neural Network Template Parameters

The primary neural network template is `NeuralNetwork`, which supports heterogeneous hidden layer sizes via the `HiddenLayers<N0, N1, ...>` variadic template. `MultilayerPerceptron` is a convenience alias for `NeuralNetwork` with uniform hidden layers.

```cpp
template<
        typename ValueType,
        size_t NumberOfInputs,
        typename HiddenLayersDescriptor,
        size_t NumberOfOutputs,
        typename TransferFunctionsPolicy,
        bool IsTrainable = true,
        size_t BatchSize = 1,
        bool HasRecurrentLayer = false,
        hiddenLayerConfiguration_e HiddenLayerConfig = NonRecurrentHiddenLayerConfig,
        size_t RecurrentConnectionDepth = 0,
        outputLayerConfiguration_e OutputLayerConfiguration = FeedForwardOutputLayerConfiguration
        >
class NeuralNetwork
{
...
```

**ValueType** - The underlying value type used by the neural network. It could be: int, float, or some other user-defined type like a QValue type. Any type will work as long as it supports the required operators.

**NumberOfInputs** - Number of input neurons in the neural network.

**HiddenLayersDescriptor** - A `HiddenLayers<N0, N1, ...>` type specifying the number of neurons in each hidden layer. Each layer can have a different size (e.g., `HiddenLayers<16, 8, 4>`). For uniform layers, the `MultilayerPerceptron` alias accepts the traditional `NumberOfHiddenLayers` and `NumberOfNeuronsInHiddenLayers` scalar parameters.

**NumberOfOutputs** - Number of output neurons in the neural network.

**TransferFunctionsPolicy** - Policy class which provides certain functionality the neural network needs: Random number generation, initial weight values, neural network activation functions, etc.

**IsTrainable** - Compile time flag to indicate whether or not the neural network is trainable. Non-trainable neural networks consume less space as code and data required for trainable neural networks is not compiled into the binary image. Tinymind neural networks can have their weights, biases, etc. loaded from the outside world so that non-trainable neural networks can have their values initialized to trained ones. Training neural networks offline and instantiating untrainable neural networks within the embedded device can save a lot of code and data space.

**BatchSize** - Batch size for trainable neural networks who want to accumulate a BatchSize amount of samples before back-propagating the error thru the network.

**HasRecurrentLayer** - Compile time flag which configures the neural network's hidden as either purely feed-forward or recurrent.

**HiddenLayerConfig** - Hidden layer configuration parameter to choose between: Feed-forward, simple recurrent, GRU, or LSTM hidden layer type.

**RecurrentConnectionDepth** - Recurrent connection depth(i.e. number of backwards time steps to save in the recurrent layer).

**OutputLayerConfiguration** - Output layer configuration which configures the neural network's output layer as either feed forward or classifier type.

### MultilayerPerceptron Alias

`MultilayerPerceptron` is a backward-compatible alias for `NeuralNetwork` with uniform hidden layers:

```cpp
// These two are equivalent:
typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 3, 1, TF> MyNN;
typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<3>, 1, TF> MyNN;
```

### Usage Examples

```cpp
// Single hidden layer with 5 neurons
typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<5>, 1,
    TransferFunctionsType> SingleLayerNN;

// Two hidden layers: 8 neurons then 4 neurons
typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<8, 4>, 1,
    TransferFunctionsType> TwoLayerNN;

// Three hidden layers: 16 -> 8 -> 4
typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<16, 8, 4>, 1,
    TransferFunctionsType> ThreeLayerNN;
```

## Neural Network Class Diagram

The core class for all tinymind neural networks is `NeuralNetwork`. This class is configured via the template parameters to have the desired behavior. Some instances of the template parameters are instantiated within the class, while others are simply used by the class via static function calls. A simple class diagram of `NeuralNetwork` and its relationship to others is presented below.

![nn_class](https://user-images.githubusercontent.com/1591721/200402130-d9ba68d1-35f5-4d77-a0ca-b479cf93e059.png)

## Neural Network System Sequence Diagrams

### Initialization

When the neural network is initialized, it has to configure itself by calling upon its TransferFunctionsPolicy. From its TransferFunctionsPolicy it gets initial weight values, initial learning rate, etc.

![nn_ssd_init](https://user-images.githubusercontent.com/1591721/200402172-ac715e24-bc35-49e2-8653-98c292c09e10.png)

Tinymind does not assume anything about the environment within which the neural network is instantiated. This is why it relies upon the presence of a policy class, TransferFunctionsPolicy. This makes the code more portable in that is does assume the presence of random number generation hardware. It also does not initialize the network itself. It relies upon the policy class to handle that in any way it desires.

### Training

The simple training flow thru a feed-forward neural network is documented here.

![nn_ssd_train_pt1](https://user-images.githubusercontent.com/1591721/200402214-fc36f88f-e37f-40f4-850a-861f414dc232.png)

In the feed-forward pass, inputs are fed into the neural network. The neural network calculates the predicted output values and stores them in the OutputLayer. When `calculateError` is called, the error between the predicted output and the actual output fed into the neural network is returned to the caller. If the delta between the neural network calculated outputs and known outputs is too large, the creator of the neural network can call the `trainNetwork` API to force the neural network to back-propagate the calculated error thru the network layers and update the connection weights to try and minimize this error on the next invocation.

`NeuralNetwork` is also the base class for the recurrent network templates: `LstmNeuralNetwork`, `GruNeuralNetwork`, `ElmanNeuralNetwork`, and `RecurrentNeuralNetwork`. See the [LSTM and GRU Recurrent Networks]({{ site.baseurl }}/architectures/lstm-gru) page for details.

# Training Policies

The `TransferFunctionsPolicy` template parameter now supports additional optional policies for controlling the training process. These are extracted via SFINAE traits, so existing code that doesn't use them compiles unchanged:

```cpp
typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGeneratorPolicy,          // weight initialization RNG
    HiddenNeuronActivationPolicy,         // e.g. TanhActivationPolicy
    OutputNeuronActivationPolicy,         // e.g. SigmoidActivationPolicy
    NumberOfOutputNeurons,                // default: 1
    NetworkInitializationPolicy,          // default: DefaultNetworkInitializer
    ErrorCalculatorPolicy,                // default: MeanSquaredErrorCalculator
    ZeroTolerancePolicy,                  // default: ZeroToleranceCalculator
    GradientClippingPolicy,               // default: NullGradientClippingPolicy
    WeightDecayPolicy,                    // default: NullWeightDecayPolicy
    LearningRateSchedulePolicy,           // default: FixedLearningRatePolicy
    OptimizerPolicy                       // default: NullOptimizerPolicy (SGD)
> TransferFunctionsType;
```

The new training policies include:
- **Gradient clipping** (`GradientClipByValue`) -- clamps gradients to prevent overflow
- **L2 weight decay** (`L2WeightDecay`) -- ridge regularization
- **Learning rate scheduling** (`StepDecaySchedule`) -- multiplicative decay at intervals
- **Optimizers** (`AdamOptimizer`, `RmsPropOptimizer`) -- adaptive per-parameter learning rates

See the [Advanced Training Techniques]({{ site.baseurl }}/training/training-techniques) page for detailed documentation and examples.

# Activation Functions

Tinymind supports the following activation functions, all available for both fixed-point (via lookup tables) and floating-point:

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

ELU uses the exp lookup table. GELU approximates `x * sigmoid(1.702 * x)` using the existing sigmoid lookup table. SiLU computes `x * sigmoid(x)` and is used internally by KAN edge functions.

# Additional Network Types

Beyond the feed-forward and recurrent architectures described above, tinymind also provides:

- **[Kolmogorov-Arnold Networks]({{ site.baseurl }}/architectures/kan)** -- learnable B-spline activation functions on edges
- **[Convolutional and Pooling Layers]({{ site.baseurl }}/architectures/conv-pooling)** -- Conv1D, MaxPool1D, AvgPool1D, BatchNorm1D, Dropout
- **[Quantized Neural Networks]({{ site.baseurl }}/architectures/quantized-networks)** -- BinaryDense and TernaryDense layers
