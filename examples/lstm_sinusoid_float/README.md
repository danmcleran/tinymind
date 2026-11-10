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
- plain SGD at a learning rate of 0.2 (no momentum, no acceleration), training
  on every step
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

`make run` also prints the RMS error of each curve against the true sinusoid
(seed 7):

- **one-step-ahead:** double 0.0058, Q16.16 0.0044
- **free-run:** double 0.0142, Q16.16 0.0116

Both precisions lie on the sinusoid in both modes, and the free run holds
amplitude and phase through both periods. Across seeds 1 to 8 the one-step RMS
error stays between 0.003 and 0.018 for both, so at this size the Q16.16
quantization cost is within the seed-to-seed spread. (Neither output reaches
exactly 0 or 1: the sigmoid saturates before the rails, an activation artifact
shared by both paths.)

Two training details matter. The example trains on every step: an earlier
version skipped any step whose error was within the zero tolerance, which
stopped training once the hidden-to-output readout alone fit the stream to that
tolerance, before the LSTM gates had learned the dynamics (one-step RMS 0.044
double and 0.057 Q16.16, free-run 0.053 and 0.27). And it uses no momentum.
With momentum 0.5 and acceleration 0.0625 the Q16.16 network used to diverge
when trained on every step: the fixed-point momentum product rounded a one-LSB
delta weight back to itself, so every weight drifted by an LSB per step whatever
the gradient. The momentum terms now round toward zero and the network
converges with momentum as well, but at this learning rate it ends less
accurate than plain SGD (Q16.16 one-step RMS 0.017 against 0.0044).

The matplotlib style comes from the shared
[`examples/plotting/tinymind_plot.py`](../plotting) module; install it into an
isolated environment (venv/pyenv), never system Python.
