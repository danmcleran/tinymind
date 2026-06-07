# Gas Sensor Array Drift Classifier

Six-way gas classifier that demonstrates **sensor drift** — the slow accuracy
decay a fixed model suffers as metal-oxide gas sensors age. Inspired by the UCI
[Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/224)
(16 chemo-resistive sensors × 8 features = 128 dimensions, recorded over 36
months in 10 batches), built on a Q16.16 fixed-point MLP from TinyMind.

## The drift problem

Metal-oxide gas sensors physically age: their baseline resistance and gain
slowly shift over weeks and months. A classifier trained on fresh sensor data
gradually loses accuracy on later measurements even though the *gases* are
identical — the **sensor response distribution** has moved out from under the
model. This is the central challenge the UCI dataset was created to study.

This example reproduces that phenomenon and makes it visible as a **drift
curve**: train once on the first batch, then watch accuracy fall batch by batch.

## Synthetic data

The real dataset is large and multi-file. Following the
`examples/predictive_maintenance` precedent, this example **synthesizes** a
physically-motivated dataset with documented rules (see `synthesizeDataset()`):

- **6 gas classes** (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone,
  Toluene). Each has a distinct 128-dim mean response vector (a per-sensor gain
  profile × a per-feature profile) so the classes are cleanly separable, plus
  per-sample Gaussian noise.
- **10 sequential batches** (months). Batch 1 is drift-free. Each later batch
  applies a per-sensor-feature **multiplicative gain** and **additive offset**
  whose magnitude grows linearly with the batch index — a simple, documented
  stand-in for real sensor aging.
- Deterministic: `std::srand(7U)`, 150 samples/class/batch.

## Network

- 128 inputs (16 sensors × 8 features), z-score normalized using **batch-1
  training statistics** then scaled by 1/3 to sit inside Q16.16's stable range.
  The same batch-1 statistics are applied to every batch — normalization does
  **not** compensate for drift, which is the point.
- 1 hidden layer of 32 ReLU neurons
- 6 sigmoid outputs — one-hot encoding of the gas class; predicted class is argmax
- 50k iterations of uniform random sampling from the **batch-1** 80% train split
- Evaluated on each batch 1..10 to produce the drift curve

## Build and run

```bash
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

The dataset is synthetic, so there is no file to copy — `make run` cd's into
`./output` and runs the binary.

## Output

- `output/gas_loss.csv` — training loss curve (iteration, avg |error|)
- `output/gas_drift.csv` — the **drift curve** (batch 1..10, accuracy)
- `output/gas_confusion.csv` — 6×6 batch-1 validation confusion matrix
- `output/gas_drift_behavior.png` — `plot.py` renders loss + drift curve + confusion

Expected (seed = 7): batch 1 trains cleanly to high in-distribution accuracy
(>90%), and the drift curve slopes clearly downward as the batch index grows —
late batches are well below batch 1.

## TinyMind capability shown

`NeuralNet<>` feed-forward MLP (`MultilayerPerceptron`) trained and run entirely
in `QValue` Q16.16 fixed-point on a 128-input classification task — the same
"train + deploy on an MCU" path as `examples/iris`, scaled up.

## Mitigating drift (where to go next)

This example only **classifies and shows** the drift; it does not correct for it.
TinyMind's relevant mitigation path is the host-side calibration tooling in
[`cpp/include/qcalibration.hpp`](../../cpp/include/qcalibration.hpp), introduced
in the project's Phase 15 work:

- **`KLDivergenceObserver`** / **`PercentileObserver`** — robust activation-range
  calibration that clips heavy-tailed / outlier responses instead of letting a
  few drifted samples blow out the quantization range. Re-calibrating on fresh
  batches keeps the int8 grid centered on the *current* sensor distribution.
- **`crossLayerEqualizeDense`** (Cross-Layer Equalization) — rebalances per-channel
  weight/activation ranges so the model is less brittle to the per-sensor gain
  shifts that drift introduces.

The deeper fix (domain adaptation / periodic re-training on recent batches) lives
above the inference layer, but robust per-batch re-calibration with the above
observers is the first, cheap line of defense on-device.
