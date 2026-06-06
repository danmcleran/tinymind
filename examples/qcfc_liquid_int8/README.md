# qcfc_liquid_int8

int8 Closed-form Continuous-time (CfC) **liquid** cell — end-to-end deployment
exemplar. The float / fixed-point liquid cells live in `cpp/cfc.hpp` and
`cpp/ltc.hpp` (see `examples/cfc_sequence`, `examples/ltc_sequence`); this is the
pure-integer counterpart built on `cpp/qcfc.hpp`.

A small CfC cell (3 inputs → 6 hidden, 8-unit backbone) is calibrated on the
host with `qcalibration.hpp`, then driven over a 16-step input sequence through
`QCfCCell::forward` — int8 weights/activations, int32 accumulators, integer
requantization, sigmoid/tanh LUTs, **no `<cmath>` on the inference path**.

Regular-sampling deployable form: the elapsed time `ts` is a calibration
constant folded into the time-gate-A requantizer and the combined time bias.
(Irregular per-step `ts` is the float `cfc.hpp` cell's domain.)

## Build / run

```bash
make run      # parity report: max-abs int8-vs-float hidden-state error (PASS if < 0.06)
make bench    # CSV cycle/byte report (per-step forward cost + footprint)
make golden   # stable int8 hidden-state byte stream (deterministic)
```

On the bundled seed: ~0.035 max-abs error vs the float reference, 368 bytes of
int8 weights + int32 biases (plus the 256-entry sigmoid/tanh LUTs, 512 bytes,
shared across the whole model).

The `--golden` mode emits a deterministic int8 byte stream suitable for an
integration-suite fixture.
