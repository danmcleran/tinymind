# PyTorch GRU Weight Export Example

This example trains a one-layer GRU in PyTorch, exports its weights in the
format `RecurrentNetworkPropertiesFileManager` reads, and checks that
tinymind's `GruNeuralNetwork` loaded from that file computes the same outputs
as PyTorch.

## Files

| File | Purpose |
|------|---------|
| `gru_export.py` | `GRUNet` (a GRU plus a sigmoid readout), XOR training, and `export_gru_weights` |
| `gru_parity.cpp` | Loads an exported file into tinymind's GRU (double or Q16.16) and runs sequences read from stdin |
| `test_gru_parity.py` | Trains, exports, runs `gru_parity` and compares with PyTorch |

## Running it

`make` builds `output/gru_parity` (C++ only, so CI builds it with every
example). The parity test needs a Python with `torch` and `numpy`; use a
virtual environment, never the system interpreter:

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy
make parity PYTHON=.venv/bin/python
```

Expected result: the double network matches PyTorch to about 5e-13 (the
12 decimal places `gru_parity` prints), Q16.16 to about 0.003 (its lookup-table
sigmoid and tanh), and tinymind classifies XOR 4/4. The test is not part of
CI, because torch is not a CI dependency.

`python gru_export.py` on its own trains the model and writes
`gru_weights_q16_16.txt` and `gru_weights_float.txt` to the current directory.

## How PyTorch's GRU maps onto tinymind's

The two libraries write the same GRU in three different ways, and the export
handles each one.

1. **Gate order.** PyTorch stores the gates as reset, update, new
   (candidate); tinymind uses update, reset, candidate. The export reorders
   them.
2. **Update-gate convention.** PyTorch computes
   `h_t = (1 - z) * n + z * h_{t-1}`, and tinymind computes
   `h_t = (1 - z') * h_{t-1} + z' * n`. So tinymind's `z'` is PyTorch's
   `1 - z = sigmoid(-a)`, and the export negates every update-gate weight and
   bias.
3. **The candidate's recurrent bias.** PyTorch's candidate is
   `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`: the reset gate scales
   `b_hn`. tinymind has one bias per gate, added outside the reset gate, so a
   nonzero `b_hn` cannot be represented. `GRUNet` sets `b_hn` to zero and masks
   its gradient, so it stays exactly zero through training (with SGD, Adam or
   RMSprop, as long as weight decay is off), and `export_gru_weights` raises
   `ValueError` for a model whose `b_hn` is not zero. The other biases of each
   gate are summed into tinymind's single bias.

Before these were handled, the exported XOR model scored 4/4 in PyTorch and
1/4 in tinymind, with outputs up to 0.999 apart.
