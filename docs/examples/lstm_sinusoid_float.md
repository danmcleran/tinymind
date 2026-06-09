---
title: LSTM Sinusoid (Float vs Q16.16)
parent: Examples
nav_order: 57
layout: default
---

# LSTM Sinusoid — Floating-Point vs Q16.16

The floating-point counterpart to [LSTM Sinusoid](lstm_sinusoid.md). The **same** LSTM, the **same** periodic training stream, and the **same** seed, topology, and epoch budget are trained twice — once with `ValueType = double`, once with `ValueType = QValue<16, 16>` — then overlaid on one chart so the quantization cost can be read off directly. This is the only example that exercises a recurrent (LSTM) network in floating-point; every other RNN exemplar in the repo is fixed-point only.

## How it works

- One input (`sin[t]` scaled to [0, 1]), a single hidden layer of 16 LSTM units, one sigmoid output — identical on both precision paths. 20 samples/period, 20000 epochs over the continuous stream `sin[t] -> sin[t+1]` with the recurrent state wrapping across the period boundary.
- Both nets are seeded identically (`srand(7)`) before construction, and both weight initializers draw from `[-0.5, 0.5]`, so they start from comparable inits — the only difference between the two runs is the numeric type.
- The library only ships `FixedPointTransferFunctions`. The floating-point transfer-function bundle and the `double` activation specializations (`TanhActivationPolicy<double>`, `SigmoidActivationPolicy<double>`, `Constants<double>`, `ZeroToleranceCalculator<double>`) at the top of the source are the same ones `unit_test/nn/nn_unit_test.cpp` uses to exercise the float path. The LSTM cell internally hardcodes sigmoid gates and tanh cell-state, so those two `double` specializations are what the recurrent path actually calls.
- **One-step-ahead** is always fed the true `sin[t]` — it measures how well the dynamics were learned. **Free-run** feeds each prediction back as the next input — the honest generation test, where small one-step errors compound into amplitude/phase drift.

## Build and run

```bash
cd examples/lstm_sinusoid_float
make release
make run       # writes output/lstm_sinusoid_float.csv
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` writes `output/lstm_sinusoid_float.csv` with columns `step,true,float_one_step,float_free_run,q_one_step,q_free_run` over a two-period horizon (the sinusoid is scaled to the [0, 1] range to match the sigmoid output).

## Output

![LSTM sinusoid: float vs Q16.16]({{ site.baseurl }}/assets/plots/lstm_sinusoid_float.png)

- **One-step-ahead (top):** both precisions track the sinusoid closely; the float curve sits a hair tighter and the Q16.16 curve lags slightly at the peaks. Neither reaches exactly 0 or 1 — the sigmoid output saturates before the rails, an activation artifact shared by both paths, not a precision effect.
- **Free-run (bottom):** the float net keeps oscillating with roughly the right amplitude through both periods, accumulating only a slow phase lag, while the Q16.16 net collapses in the second period — the coarser fixed-point grid loses the phase information the LSTM state needs to stay on the sinusoid.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/lstm_sinusoid_float)
