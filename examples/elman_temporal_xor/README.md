# Elman Network — Temporal XOR

The textbook demonstration of what a recurrent (Elman) network buys you: memory
a plain feed-forward network does not have.

A stream of random bits `x[0], x[1], ...` is fed one per timestep. The target at
step `t` is the XOR of the **current** and **previous** bit:

```
target[t] = x[t] XOR x[t-1]        (x[-1] = 0)
```

The network only ever sees `x[t]` — never `x[t-1]`. So for any given input bit the
correct answer is 0 or 1 with equal probability, depending on history a memoryless
model cannot observe. A plain MLP therefore cannot beat chance: its best response
to each input is 0.5. An [`ElmanNeuralNetwork`](../../cpp/neuralnet.hpp) has a
recurrent hidden context that can hold `x[t-1]`, so it recovers the XOR.

The same task is trained on both an `ElmanNeuralNetwork` and a feed-forward
`NeuralNetwork` of the **same** shape, in Q16.16 fixed-point, so the only variable
is the recurrent connection.

## Network

- 1 input — the current bit `x[t]`
- 1 hidden layer of 8 units (tanh); the Elman variant adds a recurrent
  connection so the hidden context from `t-1` is available at `t`
- 1 sigmoid output — the predicted `x[t] XOR x[t-1]`
- Q16.16 fixed-point (`QValue<16, 16>`) with `GradientClipByValue`
- 60 passes over a fixed 2000-bit training stream

A plain Elman/recurrent network exposes no public state reset (only the gated
`LstmNeuralNetwork` / `GruNeuralNetwork` do), so the recurrent context flows
continuously; temporal XOR only needs one step of memory, so this is harmless.
At evaluation one warm-up bit is fed before scoring to align the context with the
`prev` bit the target depends on.

## Build & run

```bash
cd examples/elman_temporal_xor
make clean && make
make run     # writes output/elman_temporal_xor.csv, prints held-out accuracy
make plot    # renders output/elman_temporal_xor.png (needs matplotlib)
```

`make run` writes `output/elman_temporal_xor.csv` with columns
`step,input,target,elman,mlp` and prints held-out accuracy over a fresh 599-bit
stream.

## Result

```
Held-out accuracy (599 bits):
  Elman (recurrent) : 98.33%
  MLP   (no memory) : 51.09%
```

In the chart the Elman output snaps to the 0/1 target while the MLP output is
pinned near 0.5 — the recurrent connection is the whole difference. The few Elman
misses are the residual ~1.8%; the task is learnable but the small fixed-point net
does not reach a perfect fit.

The matplotlib style comes from the shared
[`examples/plotting/tinymind_plot.py`](../plotting) module; install it into an
isolated environment (venv/pyenv), never system Python.
