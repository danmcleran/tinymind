---
title: int8 State-Space (S4-lite)
parent: Examples
nav_order: 65
layout: default
---

# int8 State-Space (S4-lite)

A diagonal state-space (linear-recurrent / S4-lite) layer in int8, running as a streaming sequence filter. Exercises both variants from `cpp/qssm.hpp`: `QStateSpace1D` (linear time-invariant) and `QSelectiveStateSpace1D` (input-gated).

## How it works

- Each layer runs `NumChannels` independent first-order IIR recurrences — a per-channel echo/smoothing filter with pole `a[c]`:
  - **LTI:** `s_t[c] = a[c]·s_{t-1}[c] + b[c]·x_t[c]`, `y_t[c] = c[c]·s_t[c] + d[c]·x_t[c]`.
  - **Selective:** the input drive is scaled by a per-channel hard-sigmoid gate `g_t[c] = clamp(wg[c]·x_t[c] + bg[c], 0, 1)` — the content-dependent selectivity a Mamba-style model adds on top of a fixed linear recurrence, kept LUT-free.
- **O(1) streaming state.** The decode state is a fixed `NumChannels`-wide int32 vector (`QSSMState`), constant in the sequence length — the same size after 10,000 timesteps as after one. The layer streams an arbitrarily long signal in O(C) memory and work per step, which is what makes diagonal SSMs attractive for always-on sensor / audio on an MCU.
- **Two entry points, one kernel.** `step()` advances one timestep (the streaming primitive); `forward()` resets the state and runs `step()` across a block. The driver asserts the int8 `step()` decode reproduces the full-sequence `forward()` **byte-for-byte** for both variants.
- All integer at runtime, freestanding-safe: coefficients are per-channel `(multiplier, shift)` pairs (built host-side by `buildQSSMParams` / `buildQSelectiveGateParams`), and the selective gate is a clamped affine (no LUT). `|a[c]| < 1` keeps the recurrence and the int32 state bounded.
- ~0.8% (LTI) and ~0.7% (selective) max-abs error vs the float reference on the bundled signal.

## Build and run

```bash
cd examples/state_space_int8
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host calibration). `make golden` writes the int8 outputs to `output/state_space_int8.golden`.

## Output

![int8 vs float parity for the state-space layer]({{ site.baseurl }}/assets/plots/state_space_int8.png)

Left panel overlays channel 0's LTI output (the IIR-filtered waveform) — float reference vs int8-dequantized, nearly indistinguishable across all 32 timesteps. Right panel shows the per-element absolute error, all under ~0.013, consistent with the ~0.8% of range for this int8 recurrence.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/state_space_int8)
