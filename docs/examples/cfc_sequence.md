---
title: CfC Sequence (Closed-form Continuous-time)
parent: Examples
nav_order: 21
layout: default
---

# CfC Sequence (Closed-form Continuous-time)

Trains a Closed-form Continuous-time (CfC) cell plus a linear readout on irregularly sampled sequences where the elapsed time between samples carries information the cell can only get through its time-gate, and shows on held-out data that the cell uses it.

## How it works

- A single `CfCCell<1, 8, 16>` (one input, eight state neurons, 16-unit backbone) from `cpp/cfc.hpp` followed by an 8-weight + bias linear readout. Inference runs in `double`; weights are a flat caller-owned array. Float build (`TINYMIND_ENABLE_FLOAT=1`, `TINYMIND_ENABLE_STD=1`).
- Demonstrates the solver-free CfC cell: a backbone trunk, two tanh heads, and a time-gated interpolation between them, with no inner ODE iteration per step.
- The task is a first-order system `dy/dt = (u - y) / tau` driven by a piecewise-constant input `u` and observed at irregular instants. The elapsed time `ts` between samples is drawn log-uniformly from 0.1 to 2.5 (`tau = 1`), so how far `y` moves toward the new input in one step ranges from 10% to 92%. The cell's only input is `u`; `ts` reaches it only through the time-gate. Training and held-out sets are 10 sequences of 24 steps each, with independent inputs and time steps.
- Two models train with the same seed and budget (2500 epochs of `pinn::sgdStepReverse` with momentum): the full cell, and the same cell with the time-gate's `ts` coefficient (`W_A`, `b_A`) held at 0, which makes it blind to `ts`. The same scalar-templated `step<S>` supplies the `RevVar` training gradient and the `double` inference pass.

| Model | Training MSE | Held-out MSE |
|---|---|---|
| CfC, `ts` into the time-gate | 7.3e-4 | 4.5e-3 |
| CfC, `ts`-blind (`W_A = b_A = 0`) | 3.0e-2 | 5.1e-2 |
| Predict the held-out mean | | 3.2e-1 |

The `ts`-blind cell is 11.4x worse on held-out data. The example exits non-zero unless the full cell beats the `ts`-blind cell by at least 2x on held-out MSE; with `ts` replaced by a constant inside `cfc.hpp` both models score the same and it fails.

## Build and run

```bash
cd examples/cfc_sequence
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` writes `cfc_loss.csv` (per-epoch training loss of the full cell) and `cfc_fit.csv` (the first held-out sequence: time, `ts`, input, target, and both models' predictions), which `plot.py` reads. Reverse-mode training is gated by the example's `-DTINYMIND_CFC_REVERSE_TRAINING=1`.

## Output

![CfC training loss and irregularly-sampled target fit]({{ site.baseurl }}/assets/plots/cfc_behavior.png)

The left panel shows the full cell's training loss. The right panel is a held-out sequence: the full cell (orange) follows the target through steps of very different length, while the `ts`-blind cell (green) applies roughly the same response per step and overshoots or lags wherever the time step is unusually short or long.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/cfc_sequence)
