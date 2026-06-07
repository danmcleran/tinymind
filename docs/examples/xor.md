---
title: XOR (fixed-point MLP)
parent: Examples
nav_order: 10
layout: default
---

# XOR (fixed-point MLP)

Trains a small multilayer perceptron to learn the classic non-linearly-separable XOR function, the canonical "hello world" of neural networks. The four XOR cases are generated on the fly and the network is trained purely in fixed-point arithmetic.

## How it works

- Q8.8 signed fixed-point MLP with a 2 -> 3 -> 1 topology (`tinymind::MultilayerPerceptron`), tanh hidden activation and sigmoid output.
- Demonstrates that backpropagation converges in pure Q8.8 fixed-point with no FPU: weights, activations, and gradients are all `QValue<8, 8>`.
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

The curve plots the mean absolute error over the four XOR cases against training iteration. It sits near the symmetric-guess plateau early on, breaks through the 0.25 partial-solution level, and snaps to essentially zero error once the Q8.8 network has fully separated the XOR pattern, confirming convergence in fixed point.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/xor)
