---
title: Advanced Training
layout: default
parent: Training & Deployment
nav_order: 1
---

# Advanced Training Techniques

Tinymind provides several training policies that can be composed via template parameters to customize how neural networks learn. All policies are optional -- existing code that doesn't use them compiles unchanged with null/no-op defaults. Policies are extracted from the `TransferFunctionsPolicy` via SFINAE traits.

## Why Training Policies Matter for Fixed-Point

Training neural networks with fixed-point arithmetic is fundamentally harder than with floating-point. The limited dynamic range of Q-format values means that gradients, weight updates, and accumulated errors can easily overflow, producing garbage values that destroy the network's learned state. On hardware without an FPU, you have no choice but to train in fixed-point -- and without the right guardrails, training will diverge.

The training policies on this page exist specifically to make fixed-point training robust:

- **Gradient clipping** prevents a single large gradient from overflowing the Q-format range -- this is the single most important policy for fixed-point training
- **L2 weight decay** keeps weights bounded, preventing the slow drift toward overflow that accumulates over thousands of training steps
- **Learning rate scheduling** starts with larger updates (faster convergence) and reduces them over time (fine-grained precision without overflow risk)
- **Early stopping** detects convergence and halts training, saving compute cycles on battery-powered devices
- **Adam and RMSprop** provide adaptive per-parameter learning rates that naturally scale to the Q-format range, and both reuse existing connection storage so they add zero memory overhead

# Configuring Training Policies

Training policies are specified as template parameters of the `FixedPointTransferFunctions` (or floating-point equivalent) policy class:

```cpp
typedef tinymind::FixedPointTransferFunctions<
    ValueType,                                          // Q-format or float type
    RandomNumberGeneratorPolicy,                        // weight initialization RNG
    HiddenNeuronActivationPolicy,                       // e.g. TanhActivationPolicy
    OutputNeuronActivationPolicy,                       // e.g. SigmoidActivationPolicy
    NumberOfOutputNeurons,                              // default: 1
    NetworkInitializationPolicy,                        // default: DefaultNetworkInitializer
    ErrorCalculatorPolicy,                              // default: MeanSquaredErrorCalculator
    ZeroTolerancePolicy,                                // default: ZeroToleranceCalculator
    GradientClippingPolicy,                             // default: NullGradientClippingPolicy
    WeightDecayPolicy,                                  // default: NullWeightDecayPolicy
    LearningRateSchedulePolicy,                         // default: FixedLearningRatePolicy
    OptimizerPolicy                                     // default: NullOptimizerPolicy (SGD)
> TransferFunctionsType;
```

The last four parameters (gradient clipping, weight decay, learning rate schedule, and optimizer) are the new training policies. Each has a null/no-op default, so you only need to specify the ones you want.

# Adam Optimizer

Adam (Adaptive Moment Estimation) maintains per-parameter running averages of the first moment (mean) and second moment (variance) of the gradient.

## Template Declaration

```cpp
template<typename ValueType,
         int Beta1Int = 0, unsigned Beta1Frac = 230,
         int Beta2Int = 0, unsigned Beta2Frac = 255,
         int EpsilonInt = 0, unsigned EpsilonFrac = 1>
struct AdamOptimizer
```

Adam reuses the existing `mDeltaWeight` and `mPreviousDeltaWeight` storage in trainable connections, so it requires **no additional memory** beyond standard SGD.

## Example: Adam with Fixed-Point Q16.16

```cpp
typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    UniformRealRandomNumberGenerator<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    1,
    tinymind::DefaultNetworkInitializer<ValueType>,
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType>,
    tinymind::NullWeightDecayPolicy<ValueType>,
    tinymind::FixedLearningRatePolicy<ValueType>,
    tinymind::AdamOptimizer<ValueType>> TransferFunctionsType;

typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 5, 1, TransferFunctionsType> NNType;
NNType nn;

nn.setLearningRate(ValueType(0, 655)); // ~ 0.01 in Q16.16
```

## Example: Adam with Floating-Point

```cpp
typedef double ValueType;

struct AdamTF : public FloatingPointTransferFunctions<
    ValueType, RandomNumberGenerator,
    tinymind::TanhActivationPolicy,
    tinymind::TanhActivationPolicy>
{
    typedef tinymind::AdamOptimizerFloat<ValueType> OptimizerPolicyType;
};

typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 5, 1, AdamTF> NNType;
```

# RMSprop Optimizer

RMSprop maintains only the second moment (running average of squared gradients) -- it's simpler and lighter than Adam. RMSprop is often preferred for recurrent networks (LSTM, GRU).

```cpp
template<typename ValueType,
         int DecayInt = 0, unsigned DecayFrac = 230,
         int EpsilonInt = 0, unsigned EpsilonFrac = 1>
struct RmsPropOptimizer
```

# Gradient Clipping

Gradient clipping prevents exploding gradients by clamping gradient values to a fixed range. Critical for fixed-point arithmetic where large gradients can cause overflow.

```cpp
// Clip gradients to [-1.0, 1.0] (default)
typedef tinymind::GradientClipByValue<ValueType> ClipPolicy;

// Clip gradients to [-2.0, 2.0]
typedef tinymind::GradientClipByValue<ValueType, 2, 0> WiderClipPolicy;

// No clipping (null policy)
typedef tinymind::NullGradientClippingPolicy<ValueType> NoClipPolicy;
```

# L2 Weight Decay

L2 weight decay (ridge regularization) penalizes large weights by pulling them toward zero on every update: `w_new = w * (1 - lr * lambda)`.

```cpp
// Default lambda (~ 0.004 for Q8.8)
typedef tinymind::L2WeightDecay<ValueType> DecayPolicy;

// No weight decay (null policy)
typedef tinymind::NullWeightDecayPolicy<ValueType> NoDecayPolicy;
```

# Learning Rate Scheduling

Step decay reduces the learning rate by a multiplicative factor at regular intervals.

```cpp
template<typename ValueType, size_t StepInterval = 1000,
         int DecayIntegerPart = 0, unsigned DecayFractionalPart = 230>
struct StepDecaySchedule
```

```cpp
// Decay by ~0.9 every 5000 steps
typedef tinymind::StepDecaySchedule<ValueType, 5000> LRSchedule;

// Fixed learning rate (null policy)
typedef tinymind::FixedLearningRatePolicy<ValueType> FixedLR;
```

# Early Stopping

Early stopping monitors the training error and halts training when no improvement has been seen for a configurable number of steps (patience).

```cpp
tinymind::EarlyStopping<ValueType, 200> stopper;

for (int i = 0; i < 10000; ++i)
{
    nn.feedForward(&values[0]);
    error = nn.calculateError(&output[0]);

    if (stopper.shouldStop(error))
    {
        break; // no improvement for 200 steps, stop
    }

    nn.trainNetwork(&output[0]);
}
```

# Combining Policies

Here is a complete example combining gradient clipping, L2 weight decay, step decay learning rate, and Adam optimizer:

```cpp
typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    1,                                                    // NumberOfOutputNeurons
    tinymind::DefaultNetworkInitializer<ValueType>,       // initializer
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>,   // error calculator
    tinymind::ZeroToleranceCalculator<ValueType>,         // zero tolerance
    tinymind::GradientClipByValue<ValueType>,             // clip to [-1, 1]
    tinymind::L2WeightDecay<ValueType>,                   // L2 regularization
    tinymind::StepDecaySchedule<ValueType, 5000>,         // decay LR every 5000 steps
    tinymind::AdamOptimizer<ValueType>                    // Adam optimizer
> TransferFunctionsType;

typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<5>, 1,
    TransferFunctionsType> RegularizedNetwork;

RegularizedNetwork nn;
tinymind::EarlyStopping<ValueType, 500> stopper;

for (int i = 0; i < 50000; ++i)
{
    nn.feedForward(&values[0]);
    error = nn.calculateError(&output[0]);

    if (stopper.shouldStop(error))
    {
        break;
    }

    if (!TransferFunctionsType::isWithinZeroTolerance(error))
    {
        nn.trainNetwork(&output[0]);
    }
}
```

This gives you a network with:
- Gradients clamped to [-1, 1] to prevent overflow
- Weights pulled toward zero to prevent unbounded growth
- Learning rate that decays over time for fine-tuning
- Adaptive per-parameter learning rates via Adam
- Automatic convergence detection via early stopping
