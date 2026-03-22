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
