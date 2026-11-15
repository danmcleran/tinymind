---
title: LSTM Importer Round-Trip
parent: Examples
nav_order: 71
layout: default
---

# LSTM Importer Round-Trip

An end-to-end check that the [`apps/import_pytorch`](https://github.com/danmcleran/tinymind/tree/master/apps/import_pytorch) importer turns a model that starts with an `nn.LSTM` into something the C++ `cpp/qlstm.hpp::QLSTMCell` runtime reproduces.

![LSTM importer int8 parity]({{ site.baseurl }}/assets/plots/import_lstm_demo.png)

## How it works

1. `demo.py` lays out a deterministic float LSTM (3 inputs, 8 hidden units) exactly as `nn.LSTM`'s state dict, gates i, f, g, o, and adds a linear 8 → 2 head, with numpy only. It goes through `LSTM.from_state_dict`, the path a trained model takes.
2. It calibrates the model twice through `import_pytorch_model` on 128 noisy multi-sine sequences of 24 steps, once with an int16 cell state and once with an int8 one, and emits `weights_cell16.hpp` and `weights_cell8.hpp`: the activation scales (the hidden state's symmetric, zero point 0), the cell scale and range, the fixed 8/127 gate grid, per-gate int8 input and recurrent weights with their scales, the int32 gate biases, and the head's weights and biases.

   It also emits `lstm_reference.hpp`: 16 held-out sequences, the float model's outputs, and the float hidden state after every step of the first sequence. The reference comes from the LSTM equations written out separately in `demo.py`, not from the importer's own float forward, so the check cannot pass by the importer agreeing with itself.
3. `import_lstm_demo.cpp` fills `QLSTMScales` from each header, calls `buildQLSTMParams`, builds the sigmoid and tanh LUTs, runs `QLSTMCell` over each sequence from zero state and the `QDense` head on the last hidden state, and compares the dequantized output with the float reference.

## Build and run

```bash
cd examples/import_lstm_demo
make
make run                  # runs the parity check
make plot                 # output/import_lstm_demo.png (needs matplotlib)
make regenerate-pytorch   # re-runs demo.py to regenerate the headers (needs numpy)
```

## Output

```
import_lstm_demo parity test (16 sequences x 24 steps, 3-8 LSTM + 8-2 head)
  int16 cell: max |y_int8 - y_float| = 0.062568  (hidden trace 0.063370)  tolerance 0.080000 : PASS
  int8  cell: max |y_int8 - y_float| = 0.073909  (hidden trace 0.072047)  tolerance 0.080000 : PASS
```

The mean output error is 0.022 with the int16 cell and 0.035 with the int8 cell. The error is already 0.018 after the first step and does not grow over the 24 steps: it is the floor set by the fixed 8/127 gate grid and the 1/128 tanh LUT, not drift. On this 24-step sequence the int16 cell buys little; `cpp/qlstm.hpp` recommends it for long unroll horizons, where the int8 cell's 256 levels are the limit.

See [LSTM and GRU]({{ site.baseurl }}/architectures/lstm-gru#int8-quantized-counterparts) for the cell and [PyTorch → tinymind int8 (importer)]({{ site.baseurl }}/getting-started/pytorch-importer#sequence-models-one-nnlstm-first) for the importer flow.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/import_lstm_demo)
