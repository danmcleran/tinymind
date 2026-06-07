---
title: int8 QCfC Liquid Cell
parent: Examples
nav_order: 22
layout: default
---

# int8 QCfC Liquid Cell

A pure-integer int8 deployment of a Closed-form Continuous-time (CfC) liquid cell, calibrated on the host and run through a recurrent forward pass with no `<cmath>` on the inference path.

## How it works

- A small `QCfCCell` (3 inputs, 6 hidden units, 8-unit backbone) from `cpp/qcfc.hpp`: int8 weights and activations, int32 accumulators, integer requantization, and sigmoid/tanh lookup tables. It is the deployable counterpart of the float `cfc.hpp` cell.
- Demonstrates the int8 quantization path (`TINYMIND_ENABLE_QUANTIZATION=1`) applied to a recurrent liquid cell: the cell is driven over a 24-step input sequence — three phase-shifted sinusoids — and its hidden trajectory is compared step by step against a float reference. The structured drive gives the state a real two-sided dynamic range, so the parity plot reads as a quantization staircase riding the float curve rather than dithering in a tiny structureless band.
- This is the regular-sampling deployable form: the elapsed time `ts` is a calibration constant folded into the time-gate-A requantizer and the combined time bias. Host-side calibration uses `qcalibration.hpp`; the runtime forward pass is integer-only.

## Build and run

```bash
cd examples/qcfc_liquid_int8
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` reports the max-abs int8-vs-float hidden-state error (PASS if under 10% of the hidden range; ~0.0045, about 1.9% of the range, on the bundled seed) and writes `qcfc_parity.csv` for `plot.py`. Two extra modes: `make bench` emits a CSV cycle/byte report, and `make golden` emits a deterministic int8 hidden-state byte stream suitable for an integration-suite fixture. The build is hosted for the parity report; the deployable shape is `QUANT=1 FLOAT=0 STD=0`.

## Output

![int8 QCfC hidden-state tracking and per-step quantization error]({{ site.baseurl }}/assets/plots/qcfc_int8_parity.png)

The left panel overlays the int8 hidden state on the float reference across the 24 steps; the int8 trace rides the smooth period-8 float curve as a quantization staircase. The right panel plots the per-step max-abs error against horizontal 1/2/4-LSB guide lines — the residual sits between two and four int8 LSBs throughout, showing the gap is purely grid-resolution (int8 quantization, LUT, and requantizer rounding combined), not a modelling or convergence failure. This separation is the point: the int8 cell preserves the recurrent dynamics; what you see is the cost of the int8 grid, nothing more.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/qcfc_liquid_int8)
