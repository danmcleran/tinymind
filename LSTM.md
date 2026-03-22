# LSTM Sinusoid Prediction: Lessons Learned

## Working Configuration

The only configuration that produces sinusoidal predictions (rather than collapsing to a constant) is:

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

This produces sinusoidal auto-regressive predictions with phase drift that worsens over ~20 steps.

## Changes That Cause Collapse to Constant Prediction

Every one of the following changes, tested individually and in combination, caused the network to stop learning sinusoidal dynamics and instead predict a fixed constant (~0.5 for sigmoid output, ~-0.5 for tanh/linear output):

### Output Activation Changes
- **Tanh output activation** with [-1, 1] data: Tanh derivative vanishes near +/-1, killing gradients at the sine wave extrema.
- **Linear output activation** with [-1, 1] data: Despite having a constant derivative of 1, the network still collapsed. The larger error range (up to 2.0 vs 1.0 with sigmoid) likely causes gradient instability.

### Training Data Changes
- **Increasing samples per period** (20, 40, or 50 points instead of 10): More training pairs per epoch causes the LSTM state to drift too far within each epoch, preventing convergence.
- **Training on multiple periods** (30 or 150 samples covering 3 periods): Same issue as above, amplified.

### State Management Changes
- **Resetting LSTM cell state between epochs**: Prevents the network from building up useful temporal context. The accumulated state across epochs appears to act as implicit regularization that is essential for this implementation.

### Hyperparameter Changes
- **Increasing hidden neurons to 32**: Too many parameters for the small training set, causes overfitting to the mean.
- **Increasing training iterations to 200,000**: Network converges to the same result as 100,000 iterations; no improvement.
- **Lowering learning rate (0.05-0.10)**: Insufficient gradient magnitude to escape the mean-prediction basin.
- **Lowering momentum (0.2-0.3)**: Similar collapse to mean prediction.

## Root Cause Analysis

The auto-regressive phase drift is inherent to this architecture because:

1. **Only 8 unique training pairs** from 10 samples per period is too few for the LSTM to precisely learn the sinusoidal mapping.
2. **No LSTM state reset between epochs** means the cell state is an accumulation from all training, not a clean traversal of one sequence. This makes the state at prediction time inconsistent with any single training pass.
3. **Auto-regressive error compounding**: Each predicted value feeds back as input. Small errors in amplitude or phase accumulate, causing the predicted wave to drift relative to the true wave.
4. **Two inputs provide limited context**: With only (sin[i-1], sin[i]), the network must infer both phase and direction from just two values.

## Library Improvements Made

The following additions to the library are useful for future experimentation:

- **`LstmNeuralNetwork::resetState()`** — Public method to reset all LSTM hidden layer cell states to zero. Located in `cpp/neuralnet.hpp`.
- **`Layer::getNeuron(index)`** — Public accessor to retrieve a pointer to an individual neuron by index. Located in `cpp/neuralnet.hpp`.
- **`LinearActivationPolicy` derivative fix** — Corrected `activationFunctionDerivative()` to return 1 instead of 0 (the mathematically correct derivative of f(x) = x). Located in `cpp/activationFunctions.hpp`.

## Recommendations for Future Improvement

To significantly improve auto-regressive prediction accuracy, changes deeper in the LSTM training algorithm would be needed:

1. **Teacher forcing / scheduled sampling**: During training, randomly feed the model's own prediction (instead of the true value) as input. This teaches the model to recover from its own errors.
2. **Multi-step training loss**: Unroll the auto-regressive prediction for K steps during training and compute loss on all K steps.
3. **Gradient clipping**: Prevent exploding gradients during BPTT by clipping gradient magnitudes.
4. **BPTT truncation**: Limit backpropagation through time to a fixed window to stabilize training with longer sequences.
5. **More inputs**: Using 3-4 previous values instead of 2 would give the network more context to determine phase position.

## Critical Architectural Finding

### Shared Pre-Activation Across All Gates

The root cause of TinyMind's LSTM instability is that all gates share a single pre-activation value rather than having independent weight matrices per gate.

In `cpp/neuralnet.hpp` (lines 2313-2324), the LSTM feedforward computes a single `input` value from the previous layer and recurrent connection, then passes that same value to all three gates:

```cpp
input = previousLayer.getBiasNeuronValueForOutgoingConnection(neuron);
input += previousLayer.getOutputValueForOutgoingConnection(neuron);
input += recurrentLayer.getOutputValueForOutgoingConnection(neuron);

forgetGateActivation = ForgetGateActivationPolicy::activationFunction(input + previousCellState);
inputGateActivation  = InputGateActivationPolicy::activationFunction(input);
outputGateActivation = OutputGateActivationPolicy::activationFunction(input);
```

### How a Standard LSTM Works

A standard LSTM has **4 independent weight matrices**, each with its own weights and biases. For an input `x_t` and previous hidden state `h_{t-1}`, a standard LSTM computes:

- **Forget gate**: `f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)`
- **Input gate**: `i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)`
- **Cell candidate**: `c_hat_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)`
- **Output gate**: `o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)`

Each gate learns **different** linear projections of the input and recurrent state, allowing each gate to respond to different features. TinyMind computes a single linear projection and feeds it to all gates, which means:

- The input gate, output gate, and cell candidate all see the exact same pre-activation value.
- The forget gate sees that value plus the previous cell state, but still has no independent learned weights.
- The gates cannot learn independent behaviors. For example, the forget gate cannot learn to retain information while the input gate simultaneously learns to suppress new input, because both are driven by the same projection.

### Why This Explains All Observed Collapse Behavior

Every failure mode documented above traces back to this shared-weight architecture:

1. **Narrow hyperparameter window**: With shared weights, only a very specific learning rate, momentum, and network size produce the right balance where the single shared projection happens to drive all gates in a useful way. Any change disturbs this fragile equilibrium.

2. **Collapse on LSTM state reset**: The accumulated cell state is the only source of differentiation between the forget gate and the other gates (via `input + previousCellState`). Resetting state removes this differentiation, making the forget gate behave nearly identically to the input and output gates.

3. **Collapse with more training data**: More data points per epoch push the shared projection through more updates per epoch, causing the single set of weights to be pulled in too many directions at once. Independent gate weights would allow each gate to specialize.

4. **Collapse with different activations**: Tanh and linear output activations expose the instability more directly. Sigmoid output with [0, 1] data happens to compress errors into a narrow range that masks the architectural limitation.

### Comparison: Standard LSTM vs TinyMind LSTM

| Feature | Standard LSTM | TinyMind LSTM |
|---------|---------------|---------------|
| **Gate weight matrices** | 4 independent (W_f, W_i, W_c, W_o), each with own input and recurrent weights | 1 shared pre-activation for all gates |
| **Gate biases** | Independent per gate (b_f, b_i, b_c, b_o) | Single shared bias |
| **Input format** | Sequential: one timestep per forward pass, state carried across steps | Parallel: all inputs presented simultaneously as a vector |
| **Training data** | Full sequences with BPTT through time | Standard backpropagation, no true BPTT |
| **Output activation** | Typically linear or task-specific | Sigmoid/tanh/linear (sigmoid is only stable option) |
| **Optimizer** | Adam or RMSProp (adaptive learning rates) | SGD with momentum only |
| **Gradient clipping** | Standard practice (clip by norm or value) | Not implemented |
| **Parameter count (16 hidden)** | ~4x more (independent weights per gate) | ~1/4 of standard (shared weights) |

### Prioritized Recommended Fixes

Listed in order of expected impact:

1. **Independent gate weight matrices** (highest priority): Each gate (input, forget, output, cell candidate) needs its own set of input weights, recurrent weights, and bias. This is the fundamental fix. Without it, the LSTM cannot learn to independently control information flow. Each neuron would need 4 separate weighted sums computed from the input layer and recurrent layer, rather than sharing a single `input` value.

2. **Sequential input processing**: Feed one timestep per forward pass and carry hidden/cell state across timesteps within a sequence. The current architecture feeds all inputs as a flat vector, which conflates spatial and temporal dimensions. True sequential processing is required for the LSTM to learn temporal patterns properly.

3. **Gradient clipping**: Implement gradient clipping by norm or value during backpropagation. LSTMs are prone to exploding gradients during BPTT, and without clipping, gradients can destabilize training or cause NaN values. This is especially critical for fixed-point implementations where overflow is catastrophic.

4. **Adam optimizer**: Replace SGD-with-momentum with Adam or RMSProp. Adaptive learning rate optimizers maintain per-parameter learning rates, which is particularly important when different gates need to learn at different rates. Adam also provides better convergence for recurrent networks in general.

5. **Proper BPTT implementation**: Once sequential input and independent gates are in place, implement truncated backpropagation through time so gradients flow back through the unrolled timesteps, enabling the network to learn temporal dependencies beyond a single step.
