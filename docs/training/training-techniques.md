---
title: Advanced Training
layout: default
parent: Training & Deployment
nav_order: 1
---

# Advanced Training Techniques

tinymind provides several training policies that can be composed via template parameters to customize how neural networks learn. All policies are optional -- existing code that doesn't use them compiles unchanged with null/no-op defaults. Policies are extracted from the `TransferFunctionsPolicy` via SFINAE traits.

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
         int Beta1Int = 0, unsigned long long Beta1Frac = QFractionLsbs<ValueType, 9, 10>::value,
         int Beta2Int = 0, unsigned long long Beta2Frac = QFractionLsbs<ValueType, 99, 100>::value,
         int EpsilonInt = 0, unsigned long long EpsilonFrac = QAdamEpsilonLsbs<ValueType, 100>::value>
struct AdamOptimizer
```

Each constant is an `(IntegerPart, FractionalPart)` pair for the `QValue(int, fraction)` constructor, where `FractionalPart` is a raw count of LSBs in `ValueType`'s own format (230 is 230/256 in Q8.8 but 230/65536 in Q16.16). The defaults are computed for whatever format you use: `QFractionLsbs<ValueType, Num, Den>` is Num/Den rounded to the format and kept inside (0, 1).

The fixed-point defaults are beta1 = 0.9, beta2 = 0.99 and epsilon = sqrt(LSB / (1 - beta2)): 0.039 in Q16.16, 1.5e-4 in Q32.32. They differ from the floating-point textbook values (beta2 = 0.999, epsilon = 1e-8, which `AdamOptimizerFloat` uses) because the second moment underflows in fixed point. For any |g| < sqrt(LSB / (1 - beta2)), the increment (1 - beta2) * g^2 rounds to zero, so v stops tracking small gradients and the step lr * m / (sqrt(v) + epsilon) is limited only by epsilon. An epsilon of one LSB lets that step reach hundreds of times the textbook step; an epsilon equal to the gradient size at which v underflows keeps it within about one learning rate. On the Q16.16 XOR network in `unit_test/nn`, the textbook constants reach the error target on 6 of 24 seeds and these defaults on 24 of 24. Q8.8 has too few fractional bits for Adam; use Q16.16 or wider.

The network advances Adam's timestep once per weight update (every `trainNetwork()` call with a batch size of 1, once per batch otherwise), and updates every weight and every bias with the same rule. The bias correction is kept for every step (beta^t is tracked incrementally).

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
         int DecayInt = 0, unsigned long long DecayFrac = QFractionLsbs<ValueType, 9, 10>::value,
         int EpsilonInt = 0, unsigned long long EpsilonFrac = QFractionLsbs<ValueType, 1, 100000000>::value>
struct RmsPropOptimizer
```

The defaults are decay = 0.9 and epsilon = 1e-8, rounded to the format (epsilon becomes one LSB). RMSprop's step is about the learning rate per parameter whatever the gradient's scale, so it needs a smaller learning rate than SGD (0.01 rather than SGD's default 0.15). Use `RmsPropOptimizerFloat` for `float`/`double`.

# Gradient Clipping

Gradient clipping prevents exploding gradients by clamping gradient values to a fixed, symmetric range. Critical for fixed-point arithmetic where large gradients can cause overflow. The network clips every weight and bias gradient, under SGD and under the adaptive optimizers.

```cpp
// Clip gradients to [-1.0, 1.0] (default)
typedef tinymind::GradientClipByValue<ValueType> ClipPolicy;

// Clip gradients to [-2.0, 2.0]
typedef tinymind::GradientClipByValue<ValueType, 2, 0> WiderClipPolicy;

// Clip gradients to [-1.5, 1.5] in Q16.16 (32768 LSBs = 0.5)
typedef tinymind::GradientClipByValue<ValueType, 1, 32768> FractionalClipPolicy;

// No clipping (null policy)
typedef tinymind::NullGradientClippingPolicy<ValueType> NoClipPolicy;
```

# L2 Weight Decay

L2 weight decay (ridge regularization) penalizes large weights by pulling them toward zero on every update: `w_new = w * (1 - lr * lambda)`. It applies to connection weights, not biases. With SGD this is exactly an L2 penalty in the loss. Under Adam and RMSprop the same shrink is applied before the adaptive step, which makes it decoupled weight decay (as in AdamW) rather than an L2 term added to the gradient.

```cpp
// Default lambda = 0.001, rounded to the format and at least one LSB
// (66/65536 in Q16.16; 1/256 in Q8.8, where 0.001 is below one LSB)
typedef tinymind::L2WeightDecay<ValueType> DecayPolicy;

// No weight decay (null policy)
typedef tinymind::NullWeightDecayPolicy<ValueType> NoDecayPolicy;
```

# Learning Rate Scheduling

Step decay reduces the learning rate by a multiplicative factor at regular intervals.

```cpp
template<typename ValueType, size_t StepInterval = 1000,
         int DecayIntegerPart = 0, unsigned long long DecayFractionalPart = QFractionLsbs<ValueType, 9, 10>::value>
struct StepDecaySchedule
```

```cpp
// Decay by 0.9 every 5000 steps, in any Q format
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
typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;

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
