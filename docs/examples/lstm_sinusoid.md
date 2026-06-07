---
title: LSTM Sinusoid
parent: Examples
nav_order: 13
layout: default
---

# LSTM Sinusoid

Trains an LSTM to predict the next value in a sinusoidal sequence, then runs it auto-regressively (feeding its own predictions back as input) to generate a full waveform. A genuine sequence-modeling task rather than a static classifier.

## How it works

- Q8.8 signed fixed-point LSTM network (`tinymind::LstmNeuralNetwork`) with one input, a single hidden layer of 16 LSTM units, and one output; tanh + sigmoid activations and a mean-squared-error loss with gradient clipping.
- Demonstrates sequential input processing and hidden/cell state carried across time steps: during training the net learns the mapping sin[t] -> sin[t+1] over one period sampled at 10 points.
- After training it predicts auto-regressively, seeding with the first sample and feeding each prediction back as the next input so errors can compound over the 20-step rollout.

## Build and run

```bash
cd examples/lstm_sinusoid
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` writes `output/lstm_sinusoid.csv` with columns `step,true,predicted` over the 20-step auto-regressive rollout (sinusoid scaled to the [0, 1] range to match the sigmoid output).

## Output

![LSTM auto-regressive sinusoid prediction]({{ site.baseurl }}/assets/plots/lstm_sinusoid.png)

The blue ground-truth sinusoid is overlaid with the orange auto-regressive prediction. The network tracks the right period but the limited Q8.8 precision and the compounding of errors under closed-loop feedback produce a sharper, overshooting waveform, illustrating the difficulty of long-horizon auto-regressive prediction in fixed point.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/lstm_sinusoid)
