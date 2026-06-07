---
title: LSTM Sinusoid
parent: Examples
nav_order: 13
layout: default
---

# LSTM Sinusoid

Trains an LSTM to predict the next value in a sinusoidal sequence, then evaluates it two ways: one-step-ahead (always fed the true value) and auto-regressively (fed its own predictions back). A genuine sequence-modeling task rather than a static classifier.

## How it works

- Q16.16 signed fixed-point LSTM network (`tinymind::LstmNeuralNetwork`) with one input, a single hidden layer of 16 LSTM units, and one output; tanh + sigmoid activations and a mean-squared-error loss with gradient clipping.
- A single scalar `sin[t]` is an *ambiguous* predictor of `sin[t+1]` — the rising and falling branches of the sine share y-values — so the network must use its recurrent state to track phase. Q16.16 gives that state enough precision; the coarser Q8.8 grid collapses the free-run onto the 0/1 rails.
- Trained on a continuous periodic stream (20 samples/period, the state wrapping across period boundaries). At evaluation the state is reset and primed with one true period (warm-up) so it locks onto the phase before predicting.
- **One-step-ahead** is always fed the true `sin[t]` — it measures how well the dynamics were learned. **Free-run** feeds each prediction back as the next input — the honest generation test, where small one-step errors compound into amplitude/phase drift.

## Build and run

```bash
cd examples/lstm_sinusoid
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` writes `output/lstm_sinusoid.csv` with columns `step,true,one_step,free_run` over a two-period horizon (the sinusoid is scaled to the [0, 1] range to match the sigmoid output).

## Output

![LSTM sinusoid prediction]({{ site.baseurl }}/assets/plots/lstm_sinusoid.png)

The orange one-step-ahead prediction tracks the blue ground-truth sinusoid tightly across both periods (correct phase, peaks mildly damped by the sigmoid range). The green auto-regressive free-run holds the right period for the first cycle, then drifts in phase and amplitude as closed-loop errors compound — the expected difficulty of long-horizon generation from a single scalar input.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/lstm_sinusoid)
