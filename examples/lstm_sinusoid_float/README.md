# LSTM Sinusoid — Floating-Point vs Q16.16

The floating-point counterpart to [`examples/lstm_sinusoid`](../lstm_sinusoid).
The **same** LSTM, the **same** training stream, and the **same** seed, topology,
and epoch budget are trained twice — once with `ValueType = double`, once with
`ValueType = QValue<16, 16>` — so the two precisions can be overlaid on one chart
and the quantization cost read off directly.

This is the only example that exercises a recurrent (LSTM) network in
floating-point; every other RNN exemplar in the repo is fixed-point only.

## Network

Identical on both precision paths:

- 1 input — the current sinusoid sample `sin[t]`, scaled to [0, 1]
- 1 hidden LSTM layer of 16 neurons (`HiddenLayers<16>`), tanh cell activation
- 1 sigmoid output — the predicted next sample `sin[t+1]`
- 20 samples/period, 20000 epochs over the periodic stream `sin[t] -> sin[t+1]`,
  recurrent state carried across the period boundary
- both nets are seeded identically (`srand(7)`) before construction and their
  weight initializers both draw from `[-0.5, 0.5]`, so they start from
  comparable inits

The library only ships `FixedPointTransferFunctions`. The floating-point
transfer-function bundle and the `double` activation specializations
(`TanhActivationPolicy<double>`, `SigmoidActivationPolicy<double>`,
`Constants<double>`, `ZeroToleranceCalculator<double>`) at the top of the source
are the same ones `unit_test/nn/nn_unit_test.cpp` uses to exercise the float
path; the LSTM cell internally hardcodes sigmoid gates and tanh cell-state, so
those two `double` specializations are what the recurrent path actually calls.

## Evaluation

After training, each net is evaluated over a two-period horizon in two modes,
each after a teacher-forced warm-up that locks the recurrent state onto the
phase:

- **one-step-ahead** (teacher forced) — always fed the true `sin[t]`, predicts
  `sin[t+1]`. Measures how well the dynamics were learned.
- **free-run** (auto-regressive) — fed its own previous prediction back as the
  next input. The honest generation test; small one-step errors compound into
  amplitude/phase drift.

## Build & run

```bash
cd examples/lstm_sinusoid_float
make clean && make
make run    # writes output/lstm_sinusoid_float.csv
make plot   # renders output/lstm_sinusoid_float.png (needs matplotlib)
```

`make run` writes `output/lstm_sinusoid_float.csv` with columns
`step,true,float_one_step,float_free_run,q_one_step,q_free_run`.

## What the chart shows

- **one-step-ahead:** both precisions track the sinusoid closely; the float
  curve sits a hair tighter, the Q16.16 curve lags slightly at the peaks.
  (Neither reaches exactly 0 or 1 — the sigmoid output saturates before the
  rails, an activation artifact shared by both paths, not a precision effect.)
- **free-run:** the float net keeps oscillating with roughly the right amplitude
  through both periods, accumulating only a slow phase lag, while the Q16.16 net
  collapses in the second period — the coarser fixed-point grid loses the phase
  information the LSTM state needs to stay on the sinusoid.

The matplotlib style comes from the shared
[`examples/plotting/tinymind_plot.py`](../plotting) module; install it into an
isolated environment (venv/pyenv), never system Python.
