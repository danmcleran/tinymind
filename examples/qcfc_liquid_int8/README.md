# qcfc_liquid_int8

int8 Closed-form Continuous-time (CfC) **liquid** cell — end-to-end deployment
exemplar. The float / fixed-point liquid cells live in `cpp/cfc.hpp` and
`cpp/ltc.hpp` (see `examples/cfc_sequence`, `examples/ltc_sequence`); this is the
pure-integer counterpart built on `cpp/qcfc.hpp`.

A small CfC cell (3 inputs → 6 hidden, 8-unit backbone) is calibrated on the
host with `qcalibration.hpp`, then driven over a 24-step input sequence — three
phase-shifted sinusoids — through `QCfCCell::forward`: int8 weights/activations,
int32 accumulators, integer requantization, sigmoid/tanh LUTs, **no `<cmath>` on
the inference path**. The structured drive gives the recurrent state a real
two-sided dynamic range, so the parity plot shows the int8 trajectory riding the
float curve as a quantization staircase rather than dithering in a tiny band.

Regular-sampling deployable form: the elapsed time `ts` is a calibration
constant folded into the time-gate-A requantizer and the combined time bias.
(Irregular per-step `ts` is the float `cfc.hpp` cell's domain.)

## Build / run

```bash
make run      # parity report: max-abs int8-vs-float hidden-state error (PASS if < 10% of the hidden range)
make bench    # CSV cycle/byte report (per-step forward cost + footprint)
make golden   # stable int8 hidden-state byte stream (deterministic)
```

On the bundled seed: max-abs error ~0.0045 vs the float reference — about 1.9%
of the hidden state's dynamic range, i.e. a few int8 LSBs, so the residual is
quantization-grid limited rather than a modelling gap. 368 bytes of int8 weights
+ int32 biases (plus the 256-entry sigmoid/tanh LUTs, 512 bytes, shared across
the whole model).

The `--golden` mode emits a deterministic int8 byte stream suitable for an
integration-suite fixture.
