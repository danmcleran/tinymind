---
title: GRU XOR
parent: Examples
nav_order: 12
layout: default
---

# GRU XOR

Trains a Gated Recurrent Unit (GRU) network to learn XOR, exercising TinyMind's recurrent network type on a small, well-understood task instead of a feed-forward MLP.

## How it works

- Q8.8 signed fixed-point GRU network (`tinymind::GruNeuralNetwork`) with two inputs, a single hidden layer of 5 GRU units, and one output; tanh activations and a mean-squared-error loss.
- Demonstrates the recurrent (GRU) cell family with gradient clipping (`GradientClipByValue`) and an `EarlyStopping` monitor that halts training once the error stops improving.
- Each XOR sample is presented as a fresh input, so the example serves as a smoke test of GRU forward/backward correctness in fixed point rather than a true sequence task.

## Build and run

```bash
cd examples/gru_xor
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` writes `output/gru_xor_results.csv` with one row per iteration (target, learned value, error); training stops early via the `EarlyStopping` window once convergence is detected.

## Output

![GRU XOR learning curve]({{ site.baseurl }}/assets/plots/gru_xor.png)

The raw per-iteration absolute error is noisy (the recurrent state makes individual samples jumpy), so the orange 200-sample moving average is the signal to read. It settles to a low, flat band, indicating the GRU has learned XOR and the early-stopping criterion is satisfied.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/gru_xor)
