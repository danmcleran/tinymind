---
title: XOR (fixed-point MLP)
parent: Examples
nav_order: 10
layout: default
---

# XOR (fixed-point MLP)

Trains a small multilayer perceptron to learn the classic non-linearly-separable XOR function, the canonical "hello world" of neural networks. The four XOR cases are generated on the fly and the network is trained purely in fixed-point arithmetic.

## How it works

- Q16.16 signed fixed-point MLP with a 2 -> 4 -> 1 topology (`tinymind::MultilayerPerceptron`), tanh hidden activation and sigmoid output.
- Demonstrates that backpropagation converges in pure fixed-point with no FPU: weights, activations, and gradients are all `QValue<16, 16>`. (The original Q8.8 / 3-neuron build worked but lurched through the XOR saddle-point plateau before snapping to a solution; Q16.16 with a 4-neuron hidden layer descends smoothly.)
- A deterministic validation pass over the four canonical XOR cases (0/0, 0/1, 1/0, 1/1) is logged every 100 iterations, giving a clean convergence signal instead of the noisy random-sample error.

## Build and run

```bash
cd examples/xor
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` writes the learning-curve CSV `output/xor_training.csv` and a per-iteration property dump `output/nn_fixed_xor.txt`. A `make plot-trajectory` target renders the generic per-column network-trajectory view from the property dump.

## Output

![XOR fixed-point MLP learning curve]({{ site.baseurl }}/assets/plots/xor_learning_curve.png)

The curve plots the mean absolute error over the four XOR cases against training iteration. It descends smoothly from the symmetric-guess level (~0.5) and settles within ~1000 iterations. All four patterns are then classified correctly with clear margins (predictions ≈ 0.09 / 0.91); the residual ~0.09 is just the sigmoid output not fully saturating to 0/1 under the MSE objective.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/xor)
