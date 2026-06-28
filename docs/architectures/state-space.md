---
title: State-Space (S4-lite)
layout: default
parent: Architectures
nav_order: 11
---

# State-Space (S4-lite)

> **Real-world use:** an always-on vibration monitor on a motor bearing flags a developing fault from a 4 kHz accelerometer stream. The diagonal recurrence summarizes the entire history in a fixed-size state, so the per-step cost and memory are the same after a month of runtime as after the first second — the right shape for a sensor that never stops streaming.

TinyMind provides a diagonal **state-space** layer family — a linear-recurrent
sequence model (S4-lite / linear RNN) that processes a stream one timestep at a
time with a fixed-size state. It is the structured-sequence cousin of the
liquid `LTC`/`CfC` cells and of linear attention: all three replace the
quadratic attention matrix (or a growing KV cache) with an O(1)-per-step
recurrence.

| Header | Layer | Role |
|--------|-------|------|
| `cpp/qssm.hpp` | `QSSMState` | fixed `NumChannels`-wide int32 recurrent state (freestanding POD) |
| `cpp/qssm.hpp` | `QStateSpace1D` | diagonal linear time-invariant SSM |
| `cpp/qssm.hpp` | `QSelectiveStateSpace1D` | input-gated ("selective") diagonal SSM |

## Diagonal recurrence — O(1) state, any length

A diagonal state-space model runs `NumChannels` independent first-order
recurrences (one scalar IIR per channel, no cross-channel mixing in the
recurrence). Per channel `c`, with input `x_t` and state `s_t`:

```
s_t[c] = a[c] * s_{t-1}[c] + b[c] * x_t[c]      # state update
y_t[c] = c[c] * s_t[c]     + d[c] * x_t[c]      # readout (d = optional skip)
```

The whole attended history is summarized by the scalar state `s[c]`, so the
decode state is a fixed `NumChannels`-wide int32 vector that is **constant in
the sequence length** — the same size after 10,000 timesteps as after one.
Streaming inference is therefore O(NumChannels) memory and work per token, the
property that makes diagonal SSMs (and linear attention, and the liquid cells)
the right shape for always-on audio / vibration / sensor streams on a
microcontroller. `|a[c]| < 1` keeps the recurrence — and the int32 state —
bounded.

## Selective (input-gated) variant

A fixed linear recurrence treats every input the same. The "selective" idea a
Mamba-style model adds is to make the dynamics **content-dependent**.
`QSelectiveStateSpace1D` does this cheaply: a per-channel hard-sigmoid gate
modulates the input drive,

```
g_t[c] = clamp(wg[c] * x_t[c] + bg[c], 0, 1)
s_t[c] = a[c] * s_{t-1}[c] + g_t[c] * b[c] * x_t[c]
```

so the model can choose, per input, how much of each token to write into the
state. The gate is a clamped affine (computed in Q15, no exp/sigmoid LUT), so
the selective layer stays as freestanding-safe as the LTI core — unlike a full
input-dependent transition, which would need a runtime exponential.

## One kernel, two entry points

Both layers expose the same pair of methods over one token kernel:

```cpp
// Streaming: advance one timestep, emit its row. O(C) work, O(1) extra memory.
void step(const InputType* x, State& state, OutputType* y) const;

// Reset the state and run step() across a [T x C] block.
// Byte-identical to T successive step() calls.
void forward(const InputType* seq, State& state, OutputType* out) const;
```

Because `forward()` is `state.reset()` followed by a loop of `step()`, a model
calibrated over full sequences produces the **exact same int8 stream** as the
timestep-by-timestep streaming decode at deployment.

## Calibration

Coefficients are per-channel integer `(multiplier, shift)` pairs. The host
helpers `buildQSSMParams` (a/b/c/d) and `buildQSelectiveGateParams` (gate) in
`qcalibration.hpp` decompose the real coefficients and the activation/state
scales into those pairs; `a[c]` is the dimensionless decay, `b`/`c`/`d` fold
the scale ratios. The state is stored int32 at a calibrated state scale for
head-room. See the
[`state_space_int8`]({{ site.baseurl }}/examples/state_space_int8) example for
the full calibration flow and the streaming-equivalence check.
