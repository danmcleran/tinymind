# LSTM Sinusoid Prediction: Lessons Learned

## Architecture

TinyMind's LSTM now implements the standard LSTM architecture with **4 independent gate weight matrices**. Each gate (cell candidate, input, forget, output) computes its own weighted sum from input, bias, and recurrent connections:

```cpp
// For hidden neuron n, connections are indexed as:
//   n*4 + 0 = cell candidate
//   n*4 + 1 = input gate
//   n*4 + 2 = forget gate
//   n*4 + 3 = output gate

// Each gate has independent weights:
cellCandidateInput = bias[n*4+0] + W_c * x_t + U_c * h_{t-1}
inputGateInput     = bias[n*4+1] + W_i * x_t + U_i * h_{t-1}
forgetGateInput    = bias[n*4+2] + W_f * x_t + U_f * h_{t-1}
outputGateInput    = bias[n*4+3] + W_o * x_t + U_o * h_{t-1}

// Standard LSTM equations:
c_hat_t = tanh(cellCandidateInput)
i_t     = sigmoid(inputGateInput)
f_t     = sigmoid(forgetGateInput)
o_t     = sigmoid(outputGateInput)
c_t     = f_t * c_{t-1} + i_t * c_hat_t
h_t     = o_t * tanh(c_t)
```

The backpropagation decomposes the output delta through each gate using the chain rule and stored gate activations, producing 4 independent gate deltas used for per-gate gradient computation and weight updates.

### Previous Architecture (Historical)

An earlier version of TinyMind's LSTM used a shared pre-activation value for all gates — a single weighted sum fed to all gates rather than independent projections. This caused severe training instability (narrow hyperparameter window, collapse to constant predictions under most configurations). The independent gate architecture resolved these issues.

## Working Configuration

The following configuration produces sinusoidal auto-regressive predictions with phase drift that worsens over ~20 steps:

| Parameter | Value |
|-----------|-------|
| Hidden neurons | 16 |
| Hidden layers | 1 |
| Inputs | 2 (sin[i-1], sin[i]) |
| Output activation | Sigmoid |
| Gate activation | Tanh |
| Data range | [0, 1] via (sin(x) + 1) / 2 |
| Samples per period | 10 |
| Training pairs per epoch | 8 |
| Training iterations | 100,000 |
| Learning rate | 0.15 (default) |
| Momentum | 0.5 (default) |
| LSTM state reset | None (state accumulates across epochs) |

## Changes That Cause Collapse to Constant Prediction

The following changes were tested under the old shared pre-activation architecture. With the new independent gate weights, some of these may now work. They are preserved here as a record of the original experiments and as starting points for re-evaluation.

### Output Activation Changes
- **Tanh output activation** with [-1, 1] data: Tanh derivative vanishes near +/-1, killing gradients at the sine wave extrema.
- **Linear output activation** with [-1, 1] data: Despite having a constant derivative of 1, the network still collapsed. The larger error range (up to 2.0 vs 1.0 with sigmoid) likely causes gradient instability.

### Training Data Changes
- **Increasing samples per period** (20, 40, or 50 points instead of 10): More training pairs per epoch causes the LSTM state to drift too far within each epoch, preventing convergence.
- **Training on multiple periods** (30 or 150 samples covering 3 periods): Same issue as above, amplified.

### State Management Changes
- **Resetting LSTM cell state between epochs**: Prevents the network from building up useful temporal context. The accumulated state across epochs appears to act as implicit regularization.

### Hyperparameter Changes
- **Increasing hidden neurons to 32**: Too many parameters for the small training set, causes overfitting to the mean.
- **Increasing training iterations to 200,000**: Network converges to the same result as 100,000 iterations; no improvement.
- **Lowering learning rate (0.05-0.10)**: Insufficient gradient magnitude to escape the mean-prediction basin.
- **Lowering momentum (0.2-0.3)**: Similar collapse to mean prediction.

## Root Cause Analysis

The auto-regressive phase drift is inherent to this training approach because:

1. **Only 8 unique training pairs** from 10 samples per period is too few for the LSTM to precisely learn the sinusoidal mapping.
2. **No LSTM state reset between epochs** means the cell state is an accumulation from all training, not a clean traversal of one sequence. This makes the state at prediction time inconsistent with any single training pass.
3. **Auto-regressive error compounding**: Each predicted value feeds back as input. Small errors in amplitude or phase accumulate, causing the predicted wave to drift relative to the true wave.
4. **Two inputs provide limited context**: With only (sin[i-1], sin[i]), the network must infer both phase and direction from just two values.

## Library Improvements Made

The following additions to the library address previously documented issues:

### Architectural Fixes
- **Independent gate weight matrices** — Each of the 4 LSTM gates (cell candidate, input, forget, output) now has its own set of input weights, recurrent weights, and bias via `GateConnectionCount` with a 4x connection multiplier. Located in `LstmHiddenLayer::feedForward()` in `cpp/neuralnet.hpp`.
- **Gate-aware backpropagation** — `LstmHiddenLayer::computeGateDeltas()` decomposes the output delta through the LSTM cell equations to produce 4 independent gate-input deltas. Gradient computation and weight updates use `GatedGradientDispatcher` and `GatedWeightUpdateDispatcher` to process per-gate connections correctly.

### Training Infrastructure
- **Configurable gradient clipping** — `GradientClipByValue` policy clips gradients to a configurable range, preventing exploding gradients during training. Replaces the previous hardcoded [-1, 1] clip. Located in `cpp/gradientClipping.hpp`.
- **L2 weight decay** — `L2WeightDecay` policy applies regularization to prevent weight overflow, especially important for fixed-point arithmetic. Located in `cpp/weightDecay.hpp`.
- **Learning rate scheduling** — `StepDecaySchedule` policy reduces the learning rate by a configurable factor every N training steps. Located in `cpp/learningRateSchedule.hpp`.

### API Improvements
- **`LstmNeuralNetwork::resetState()`** — Public method to reset all LSTM hidden layer cell states to zero.
- **`Layer::getNeuron(index)`** — Public accessor to retrieve a pointer to an individual neuron by index.
- **`LinearActivationPolicy` derivative fix** — Corrected `activationFunctionDerivative()` to return 1 instead of 0 (the mathematically correct derivative of f(x) = x).

## Current Status

| Feature | Status |
|---------|--------|
| Independent gate weight matrices | Implemented |
| Gate-aware backpropagation | Implemented |
| Gradient clipping | Implemented (configurable policy) |
| L2 weight decay | Implemented (configurable policy) |
| Learning rate scheduling | Implemented (step decay) |
| Multi-layer LSTM | Supported via `HiddenLayers<N0, N1, ...>` |
| GRU alternative | Implemented (3-gate, ~25% less memory) |
| Teacher forcing / scheduled sampling | Not implemented |
| Adam / RMSProp optimizer | Not implemented (SGD with momentum only) |
| Truncated BPTT | Not implemented |
| LSTM/GRU weight serialization | Not implemented |

## Recommendations for Future Improvement

To further improve auto-regressive prediction accuracy:

1. **Teacher forcing / scheduled sampling**: During training, randomly feed the model's own prediction (instead of the true value) as input. This teaches the model to recover from its own errors and directly addresses phase drift.
2. **Multi-step training loss**: Unroll the auto-regressive prediction for K steps during training and compute loss on all K steps.
3. **Adam optimizer**: Replace SGD-with-momentum with Adam or RMSProp. Adaptive per-parameter learning rates are particularly important when different gates need to learn at different rates.
4. **Truncated BPTT**: Limit backpropagation through time to a fixed window to enable learning temporal dependencies spanning multiple timesteps while keeping memory bounded.
5. **More inputs**: Using 3-4 previous values instead of 2 would give the network more context to determine phase position.
6. **LSTM/GRU weight serialization**: Extend `NetworkPropertiesFileManager` to support gated connection layouts (4x for LSTM, 3x for GRU) so trained models can be saved and loaded.
