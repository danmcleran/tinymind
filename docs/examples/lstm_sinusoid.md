---
title: LSTM Sinusoid
parent: Examples
nav_order: 13
layout: default
---

# LSTM Sinusoid

Trains an LSTM to predict the next value in a sinusoidal sequence, then evaluates it two ways: one-step-ahead (always fed the true value) and auto-regressively (fed its own predictions back). A genuine sequence-modeling task rather than a static classifier.

## How it works

- Q16.16 signed fixed-point LSTM network (`tinymind::LstmNeuralNetwork`) with one input, a single hidden layer of 16 LSTM units, and one output; tanh + sigmoid activations and a mean-squared-error loss with gradient clipping. Learning rate 0.1, no momentum, and a training step on every sample.
- A single scalar `sin[t]` is an *ambiguous* predictor of `sin[t+1]` — the rising and falling branches of the sine share y-values — so the network must use its recurrent state to track phase. Q16.16 gives that state enough precision; the coarser Q8.8 grid collapses the free-run onto the 0/1 rails.
- Trained on a continuous periodic stream (20 samples/period, the state wrapping across period boundaries).
- Evaluated on a **held-out** signal: the same sinusoid shifted by half a sample, so every input and target falls between the training samples. The state is reset and primed with one true period (warm-up) before predicting.
- **One-step-ahead** is always fed the true `sin[t]` — it measures how well the dynamics were learned. **Free-run** feeds each prediction back as the next input — the honest generation test, where small one-step errors compound into amplitude/phase drift.

## Build and run

```bash
cd examples/lstm_sinusoid
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` prints the RMS errors and writes `output/lstm_sinusoid.csv` with columns `step,true,one_step,free_run` over a two-period held-out horizon (the sinusoid is scaled to the [0, 1] range to match the sigmoid output).

## Results

Held-out signal, 40 steps, RMS error:

| Model | One-step-ahead | Free-run |
|---|---|---|
| LSTM, all weights trained | **0.0076** | **0.011** |
| LSTM, only the readout trained (input, gate and recurrent weights frozen at their initial values) | 0.044 | 0.139 |
| Persistence (predict `sin[t+1] = sin[t]`) | 0.111 | — |

Across five weight-initialization seeds the full model's one-step error was 0.0064 to 0.012 on four and 0.053 on the fifth (free-run 0.53 there), against 0.043 to 0.068 with only the readout trained.

## Output

![LSTM sinusoid prediction]({{ site.baseurl }}/assets/plots/lstm_sinusoid.png)

On the held-out signal the one-step-ahead prediction (orange) and the free-run (green) both lie on the ground truth across the two periods.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/lstm_sinusoid)
