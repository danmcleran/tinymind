---
title: LTC Sequence (Liquid Time-Constant)
parent: Examples
nav_order: 20
layout: default
---

# LTC Sequence (Liquid Time-Constant)

Trains a Liquid Time-Constant (LTC) cell plus a linear readout to reproduce the step response of a leaky integrator, using only the existing reverse-mode autodiff trainer.

## How it works

- A single `LtcCell<1, 6>` (one input, six state neurons, sigmoid synapse) from `cpp/ltc.hpp` followed by a 6-weight + bias linear readout. The scalar type is `double` for inference; weights live in a flat caller-owned parameter array. Float build (`TINYMIND_ENABLE_FLOAT=1`, `TINYMIND_ENABLE_STD=1`).
- Demonstrates the continuous-time LTC cell: a fused semi-implicit-Euler ODE step with per-neuron time constants, advanced once per input sample over a 24-step sequence.
- Because `LtcCell::step<S>` is scalar-templated, the same forward code differentiates through `RevVar` (reverse-mode autodiff) to supply the training gradient and runs as plain `double` for inference, with no hand-written LTC backprop. Training is 3000 epochs of `pinn::sgdStepReverse` with momentum, clamping per-neuron tau positive after each step.

## Build and run

```bash
cd examples/ltc_sequence
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` writes `ltc_loss.csv` (per-epoch training loss) and `ltc_fit.csv` (target vs predicted across the sequence), which `plot.py` reads. The example exits non-zero unless the loss drops at least 10x. Reverse-mode training is gated by the example's `-DTINYMIND_LTC_REVERSE_TRAINING=1`.

## Output

![LTC training loss and leaky-integrator step response fit]({{ site.baseurl }}/assets/plots/ltc_behavior.png)

The left panel shows the MSE loss falling several orders of magnitude over training as the reverse-mode gradient tunes the cell and readout. The right panel overlays the predicted output on the leaky-integrator target step response; the two curves are visually indistinguishable, confirming the autodiff-trained LTC cell fits the dynamics.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/ltc_sequence)
