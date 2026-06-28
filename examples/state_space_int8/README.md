# state_space_int8

A diagonal **state-space** (linear-recurrent / S4-lite) layer in int8, running
as a streaming sequence filter. Two variants from
[`cpp/qssm.hpp`](../../cpp/qssm.hpp):

```
QStateSpace1D           (LTI)      s_t = a*s_{t-1} + b*x_t ;  y = c*s + d*x
QSelectiveStateSpace1D  (gated)    s_t = a*s_{t-1} + g_t*b*x_t ; g_t = hardsigmoid(wg*x+bg)
```

Each layer runs `NumChannels` independent first-order IIR recurrences — a
per-channel echo/smoothing filter whose pole is `a[c]`. The **selective**
variant makes the input drive content-dependent through a cheap per-channel
hard-sigmoid gate — the selectivity a Mamba-style model adds on top of a fixed
linear recurrence, here kept LUT-free so the whole layer stays
freestanding-safe.

## O(1) streaming state

The decode state is a fixed `NumChannels`-wide int32 vector (`QSSMState`),
**constant in the sequence length** — the same size after 10,000 timesteps as
after one. So the layer streams an arbitrarily long signal in O(C) memory and
work per step, which is what makes it attractive for always-on sensor / audio
on an MCU. The driver runs the int8 layer two ways — one full-sequence
`forward()` and `step()` looped timestep-by-timestep — and asserts they are
**byte-identical** (both variants).

## Build & run

```bash
make            # debug build
make run        # int8 vs float parity + step()==forward() check (both variants)
make golden     # dump int8 outputs for regression
make plot       # channel-0 LTI output overlay (float vs int8) PNG
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host
calibration). The bundled run lands at ~0.8% (LTI) and ~0.7% (selective)
max-abs error vs float.

## Coefficients

`a` (decay, `|a| < 1` for stability), `b` (input drive), `c` (readout), `d`
(skip) per channel, plus the gate `wg`/`bg` for the selective variant, are set
at the top of the source. The host helpers `buildQSSMParams` and
`buildQSelectiveGateParams` (`qcalibration.hpp`) turn the real coefficients +
activation/state scales into the per-channel integer `(multiplier, shift)`
arrays the layer consumes — a trained model's coefficients drop in unchanged.
