# Gas Sensor Array Drift Classifier

Six-way gas classifier that demonstrates **sensor drift** — the slow accuracy
decay a fixed model suffers as metal-oxide gas sensors age. Inspired by the UCI
[Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/224)
(16 chemo-resistive sensors × 8 features = 128 dimensions, recorded over 36
months in 10 batches), built on a Q16.16 fixed-point MLP from tinymind.

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
- `output/gas_adapt.csv` — the adaptation comparison (method, batch, labelled
  samples per class, mean and standard deviation of accuracy over 10 draws,
  and batch-1 accuracy after adapting)
- `output/gas_drift_behavior.png` — `plot.py` renders loss + drift curve + confusion
- `output/gas_adapt_behavior.png` — `plot.py` renders the adaptation comparison

Expected (seed = 7): batch 1 trains cleanly to high in-distribution accuracy
(>90%), and the drift curve slopes clearly downward as the batch index grows —
late batches are well below batch 1.

## tinymind capability shown

`NeuralNet<>` feed-forward MLP (`MultilayerPerceptron`) trained and run entirely
in `QValue` Q16.16 fixed-point on a 128-input classification task — the same
"train + deploy on an MCU" path as `examples/iris`, scaled up.

## Adapting to drift on the device

After the drift curve the example adapts the shipped model to each later
batch from a few labelled samples per class drawn from that batch (1, 2, 5,
10 or 20 "shots"), scores it on the rest of the batch, and averages over 10
random draws of the shots. The trained MLP is split into a frozen backbone
(the 128 → 32 ReLU layer) and a head (the 32 → 6 layer); the split reproduces
the MLP's hidden layer bit for bit, and its predictions everywhere except where
two sigmoid outputs saturate to the same value (the head compares the logits
underneath). Seven methods are compared, each starting from the shipped model:

| Method | What changes | Labels |
|---|---|---|
| `none` | nothing | — |
| `renorm` | z-score statistics recomputed from the shots | not used |
| `renorm_all` | z-score statistics from the whole batch | not used |
| `prototype` | [`PrototypeHead`](../../cpp/prototypehead.hpp): one mean embedding per class from the shots, nearest mean wins | used |
| `head` | [`DenseHead`](../../cpp/densehead.hpp) loaded with the shipped head and trained with softmax cross-entropy (head-only fine-tuning) | used |
| `bias` | the backbone's 32 biases trained, head frozen; the gradient comes from `DenseHead::backward()` | used |
| `bias_head` | biases and head trained together | used |

The trained methods run 30 epochs of per-sample SGD at learning rate 0.05
over the shots, in Q16.16 like the rest of the example.

Batch 10 accuracy (seed 7, mean of 10 draws of the shots):

| Method | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| `none` | 0.794 | 0.793 | 0.794 | 0.794 | 0.791 |
| `renorm` | 0.889 | 0.918 | 0.923 | 0.936 | 0.938 |
| `renorm_all` | 0.937 | 0.937 | 0.936 | 0.936 | 0.937 |
| `prototype` | 0.987 | 0.994 | 0.999 | 1.000 | 1.000 |
| `head` | 0.953 | 0.984 | 0.995 | 0.997 | 0.997 |
| `bias` | 0.848 | 0.896 | 0.967 | 0.986 | 0.990 |
| `bias_head` | 0.961 | 0.982 | 0.995 | 0.997 | 0.998 |

Over data seeds 1 to 10 the order is the same: unadapted 0.766 ± 0.097,
`prototype` 0.985 ± 0.008 from one shot per class, `head` 0.936 ± 0.025 from
one shot and 0.988 ± 0.005 from five, `bias` 0.876 ± 0.039 from one shot,
`renorm_all` 0.924 ± 0.052. What the numbers say:

- **Every labelled method recovers the drift**, and re-normalizing alone does
  not: the drift flips the sign of some features' gain by batch 10, which no
  per-feature z-score undoes.
- **`PrototypeHead` is the best and cheapest** here: no gradients, no learning
  rate, one pass. The backbone's embedding still separates the gases after
  drift; they have only moved, and a class mean follows them.
- **Head-only fine-tuning beats training the backbone's biases**, and training
  both adds almost nothing over the head alone. The bias method also depends
  more on the learning rate and epoch count: at learning rate 0.02 it reaches
  0.804 from one shot, at 0.1 it reaches 0.903, while the head stays between
  0.910 and 0.959.
- **Forgetting differs.** After adapting to batch 10, the DenseHead still
  classifies batch 1 (accuracy 1.000 at seed 7), the PrototypeHead does not
  (0.77 to 0.98 depending on the shots) and re-normalizing breaks it (0.17).
  For sensor drift that is harmless, because the sensors never return to
  batch 1; for adding a user or a class it is the property that matters.

The dataset is synthetic and cleanly separable (batch 1 trains to 100%), so
treat the ranking as a demonstration on this drift model, not a general
result. Published on-device-learning work on real image and biosignal data
(TinyTL, MCUNetV3, BioTrain) finds that updating more than the last layer
gives higher accuracy than the last layer alone.

Memory: the trainable 128 → 32 → 6 MLP in Q16.16 is 73,400 bytes, the
inference-only one 20,032 bytes. Either head is under 800 bytes
(`DenseHead<Q16.16, 32, 6>` 792 bytes, `PrototypeHead<Q16.16, 32, 6>` 784
bytes), so an inference-only model plus a head adapts on the device in 28% of
the RAM that training the whole network needs.

## Other ways to mitigate drift

Host-side calibration tooling in
[`cpp/include/qcalibration.hpp`](../../cpp/include/qcalibration.hpp) helps an
int8 deployment of the same model:

- **`KLDivergenceObserver`** / **`PercentileObserver`** — robust activation-range
  calibration that clips heavy-tailed / outlier responses instead of letting a
  few drifted samples blow out the quantization range. Re-calibrating on fresh
  batches keeps the int8 grid centered on the *current* sensor distribution.
- **`crossLayerEqualizeDense`** (Cross-Layer Equalization) — rebalances per-channel
  weight/activation ranges so the model is less brittle to the per-sensor gain
  shifts that drift introduces.
