---
title: GRU XOR
parent: Examples
nav_order: 12
layout: default
---

# GRU XOR

Trains a Gated Recurrent Unit (GRU) network to learn XOR, exercising tinymind's recurrent network type on a small, well-understood task instead of a feed-forward MLP.

## How it works

- Q16.16 signed fixed-point GRU network (`tinymind::GruNeuralNetwork`) with two inputs, a single hidden layer of 8 GRU units, and one output; tanh hidden + sigmoid output, mean-squared-error loss, and `GradientClipByValue` for recurrent stability.
- XOR is *not* a sequential task — each `(x, y) -> z` pair is independent — so the recurrent state is reset (`resetState()`) before every sample. Without the reset, state left over from the previous random pattern pollutes the prediction and the network never converges (it sticks near a constant 0.5 output).
- The learning curve is the deterministic 4-case XOR error (state reset before each case), sampled through training — a clean convergence signal rather than the noisy per-sample error. Learning rate / momentum (0.5 / 0.9) are tuned for this small recurrent task and overridable on the command line.

## Build and run

```bash
cd examples/gru_xor
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` writes `output/gru_xor_results.csv` (`iteration, xor_error`) and the decision-surface CSV `output/gru_xor_decision_surface.csv` (a 41×41 sweep of the trained GRU over the input grid), and prints a per-pattern report. You can override the optimizer with `./output/gru_xor <learningRate> <momentum>`.

## Output

![GRU XOR learning curve]({{ site.baseurl }}/assets/plots/gru_xor.png)

The 4-case XOR error falls smoothly from ~0.5 (chance) to ~0.07 and plateaus. All four patterns are classified correctly with clear margins (predictions between 0.05 and 0.08 for the 0 cases, 0.93 for both 1 cases); the residual is the sigmoid output not fully saturating to 0/1 at this network size and fixed-point precision.

![GRU XOR decision surface, Q16.16 fixed-point]({{ site.baseurl }}/assets/plots/gru_xor_decision_surface.png)

Sweeping the trained GRU over the `[0,1]²` input grid — resetting the recurrent state at each point — gives the predicted-vs-actual view: the heatmap is `P(XOR=1)`, the line is the 0.5 boundary, and the four corner markers are the ground-truth targets (red = 1, blue = 0). The surface is the clean XOR saddle, with each corner firmly inside the matching region — the same presentation as the [fixed-point]({{ site.baseurl }}/examples/xor.html) and [PyTorch]({{ site.baseurl }}/examples/pytorch_xor.html) XOR examples.

## What this example does and does not show

Because the state is reset before every sample, h<sub>t-1</sub> is always zero here: the GRU's recurrent weights never affect the output, and the network behaves as a gated feed-forward net. The example exercises GRU training mechanics on a task small enough to check by hand, not recurrence. It is also easy for the gates: with the input, gate and recurrent weights frozen at their random initial values and only the hidden-to-output weights trained, the GRU still classifies all four cases (four-case error 0.088, against 0.061 when everything trains, both measured before the fixed-point momentum terms were made to round toward zero; fully trained, the error is now 0.066). For tasks that only a trained recurrent state can solve, see [Elman temporal XOR]({{ site.baseurl }}/examples/elman_temporal_xor.html) and [human activity recognition]({{ site.baseurl }}/examples/har_activity.html).

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/gru_xor)
