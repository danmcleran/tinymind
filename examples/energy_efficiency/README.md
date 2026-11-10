# Energy Efficiency Regression

Two-output building energy-load regressor trained on the
[UCI Energy Efficiency dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
using a Q16.16 fixed-point MLP from tinymind. This is the library's smallest
end-to-end **regression** example (linear output), the counterpart to the
classification path in `examples/iris`.

## Network

- 8 inputs — building design features X1..X8 (relative compactness, surface
  area, wall area, roof area, overall height, orientation, glazing area,
  glazing distribution), z-score normalized (training-set statistics) then
  scaled by 1/3 to sit inside Q16.16's stable range
- 1 hidden layer of 16 ReLU neurons
- 2 **linear** outputs — Y1 = heating load, Y2 = cooling load
- Targets are standardized to z-score (then scaled by 1/6) for training; the
  linear output trains in that compact domain and predictions are
  de-standardized back to real load units before the metrics and prediction CSV
- 40k iterations of uniform random sampling from the 80% training split,
  learning rate 0.02, momentum and acceleration disabled

### Regression gotchas (fixed-point linear output)

Two things have to be handled for a stable Q16.16 regression fit, both visible
in `energy_efficiency.cpp`:

1. **Linear derivative (library fix).** Building this example surfaced a bug in
   `tinymind::LinearActivationPolicy`: its derivative used to `return 1`, which
   the single-arg QValue constructor reads as the raw field `1/2^frac`
   (~0.0000153 in Q16.16), not `1.0`. That near-zero derivative starved the
   output-layer weight updates so a fixed-point regression head could never fit
   the target mean. It now correctly returns `Constants<ValueType>::one()`
   (`cpp/activationFunctions.hpp`), so this example uses the stock policy
   directly — no local workaround needed.
2. **Bounded, momentum-free SGD.** A linear head has an unbounded gradient, so
   the example plugs `GradientClipByValue` (the transfer-function policy's 9th
   template argument, clamp to [-1, 1]) into the network and disables momentum
   and acceleration. The default nonzero momentum/acceleration terms accumulate
   independently of the learning rate and slowly destabilize the regressor.

## Build and run

```bash
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

`make run` cd's into `./output`, so the Makefile copies the bundled
`ENB2012_data.csv` there first. The full 768-row dataset ships with the example
(~35 KB).

## Output

- `output/energy_loss.csv` — training loss curve (iteration, avg |error| in the
  standardized target domain)
- `output/energy_pred.csv` — per-test-sample true/predicted heating + cooling
  load, in real load units
- `output/energy_pred_behavior.png` — `plot.py` renders the loss curve plus a
  predicted-vs-actual scatter (with a y=x reference line) for each load, with
  MAE and R² in the subplot titles

Expected (seed = 7): heating-load **R² ≈ 0.96** (MAE ≈ 1.7), cooling-load
**R² ≈ 0.92** (MAE ≈ 1.9) on the held-out 20%. For scale, ordinary least
squares on the same eight features and split reaches R² 0.92 / 0.89 (MAE 2.08 /
2.21), so the hidden layer is earning its keep.

The example trains on every sample. An earlier version skipped any sample whose
error was within the fixed-point zero tolerance, which at this target scaling
meant every residual under about 5 kWh/m², most of the data once the fit was
close; it reached only R² 0.90 / 0.88, worse than the straight-line fit.

## tinymind capability shown

`NeuralNet<>` feed-forward MLP (`MultilayerPerceptron`) trained and run entirely
in `QValue` Q16.16 fixed-point with a **linear output activation** — the
regression analogue of the sigmoid/argmax classifier in `examples/iris`. Note
the `FixedPointTransferFunctions` 5th template argument (`NumberOfOutputNeurons`)
must be set to the output count (2) for a multi-output network.
