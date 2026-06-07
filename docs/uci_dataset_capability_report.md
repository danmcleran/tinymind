---
title: UCI Dataset Capability Report
layout: default
nav_order: 11
---

# UCI Datasets → TinyMind Examples

**Scope:** the TinyMind example programs that are built around a UCI Machine
Learning Repository dataset — what each one does, which dataset it uses, and
how the data is sourced. This is the *as-built* record (one entry per runnable
example under `examples/`), not a survey of what *could* be built.

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets).

---

## Data-source convention

Each example states one of three data modes:

- **Ships real data** — the UCI CSV is bundled in the example dir and copied
  into `./output/` by the Makefile; nothing to download.
- **Optional real data** — runs offline on a deterministic synthetic series by
  default; drop the named UCI file into `./output/` and the loader picks it up.
- **Synthetic, UCI-inspired** — reproduces the documented generative/failure
  rules of the UCI dataset; no real-data loader (the dataset's *phenomenon* is
  the subject, not its exact rows).

Every example is the `NeuralNet<>` / `LstmNeuralNetwork<>` train-and-deploy
path run entirely in `QValue` **Q16.16 fixed-point**, the on-MCU shape.

---

## Examples

### Iris — species classifier
- **Dataset:** [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) — 150 inst · 4 features · 3-class.
- **Model:** MLP 4→8→3, ReLU hidden, 3 sigmoid (argmax). z-score/3 input scaling, 30k iters.
- **Data:** ships real data (`iris.data`, ~4 KB).
- **Result:** 100% test accuracy (30/30). The smallest end-to-end fixed-point classifier.
- [`examples/iris`](https://github.com/danmcleran/tinymind/tree/master/examples/iris) · [page]({{ site.baseurl }}/examples/iris)

### Energy Efficiency — building-load regression
- **Dataset:** [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) — 768 inst · 8 features · 2-target regression.
- **Model:** MLP 8→16→2, ReLU hidden, **2 linear** outputs (heating + cooling load). `LinearActivationPolicy` + `GradientClipByValue`.
- **Data:** ships real data (`ENB2012_data.csv`, ~35 KB).
- **Result:** heating R² ≈ 0.90, cooling R² ≈ 0.88. TinyMind's smallest regression example.
- [`examples/energy_efficiency`](https://github.com/danmcleran/tinymind/tree/master/examples/energy_efficiency) · [page]({{ site.baseurl }}/examples/energy_efficiency)

### Optical Handwritten Digits — 8×8 image classifier
- **Dataset:** [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) — 8×8 bitmaps (64 px, 0..16) · 10-class.
- **Model:** MLP 64→32→10, ReLU hidden, 10 sigmoid (argmax). Per-pixel z-score with constant-zero guard, 60k iters.
- **Data:** ships real data (`optdigits.tra` 3823 rows + `optdigits.tes` 1797 rows).
- **Result:** ~96% test accuracy (1729/1797). Real 64-feature image task in fixed-point.
- [`examples/optical_digits`](https://github.com/danmcleran/tinymind/tree/master/examples/optical_digits) · [page]({{ site.baseurl }}/examples/optical_digits)

### Predictive Maintenance — binary failure classifier
- **Dataset:** [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — milling-machine readings · binary (failure / no-failure).
- **Model:** MLP 10→24→1, ReLU hidden, single sigmoid. 5 process features + **3 physics product features** (power, overstrain, temp gap) + 2-dim variant one-hot. 50/50 balanced sampling for the ~3.4% failure rate.
- **Data:** optional real data (`ai4i2020.csv`); else synthesizes 10k rows from the documented HDF/PWF/OSF/TWF/RNF rules.
- **Result:** recall ~0.89, precision ~0.80, F1 ~0.84.
- [`examples/predictive_maintenance`](https://github.com/danmcleran/tinymind/tree/master/examples/predictive_maintenance) · [page]({{ site.baseurl }}/examples/predictive_maintenance)

### Human Activity Recognition — recurrent (LSTM)
- **Dataset:** [UCI HAR Using Smartphones](https://archive.ics.uci.edu/dataset/240) — tri-axial accelerometer · activity classes.
- **Model:** Q16.16 **LSTM** 3→16 (tanh)→4 sigmoid, stateful per 32-step window, argmax at final step. Classes: WALKING / WALKING_UPSTAIRS / SITTING / STANDING.
- **Data:** optional real data (long-format `har.csv`); else synthesizes physically-motivated accelerometer windows.
- **Result:** ~97.5% test accuracy (195/200). `LstmNeuralNetwork<>` for sequence classification.
- [`examples/har_activity`](https://github.com/danmcleran/tinymind/tree/master/examples/har_activity) · [page]({{ site.baseurl }}/examples/har_activity)

### Air Quality Forecasting — recurrent (LSTM)
- **Dataset:** [UCI Air Quality](https://archive.ics.uci.edu/dataset/360) — hourly pollutant series.
- **Model:** Q16.16 **LSTM** 1→16 (tanh)→1 sigmoid, next-hour forecaster. 24-hour state reset per BPTT segment, series normalized to [0.1, 0.9].
- **Data:** optional real data (`AirQualityUCI.csv`, `CO(GT)` column, `;`-separated, decimal comma, `-200` = missing); else a synthetic daily-cycle CO series.
- **Result:** one-step-ahead MAE ≈ 0.13 over a 1.3–3.0 range.
- [`examples/air_quality`](https://github.com/danmcleran/tinymind/tree/master/examples/air_quality) · [page]({{ site.baseurl }}/examples/air_quality)

### Gas Sensor Array Drift — drift demonstration
- **Dataset:** [UCI Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224) — 16 chemo-resistive sensors × 8 features, 36 months / 10 batches · 6-gas classification.
- **Model:** MLP 128→32→6, ReLU hidden, 6 sigmoid (argmax). Train on batch 1, evaluate every later batch; normalization fixed to batch-1 stats (deliberately *not* drift-corrected).
- **Data:** synthetic, UCI-inspired (per-batch multiplicative gain + additive offset stand in for sensor aging).
- **Result:** accuracy decays from ~1.0 (batch 1) to ~0.73 (batch 10) — the drift curve is the point.
- [`examples/gas_sensor_drift`](https://github.com/danmcleran/tinymind/tree/master/examples/gas_sensor_drift) · [page]({{ site.baseurl }}/examples/gas_sensor_drift)

---

## Summary

| Example | UCI dataset | Task | Capability | Data mode |
|---|---|---|---|---|
| Iris | Iris | 3-class | `NeuralNet<>` MLP | ships real |
| Energy Efficiency | Energy Efficiency | 2-target regression | MLP, linear readout | ships real |
| Optical Digits | Optical Digits (8×8) | 10-class image | MLP | ships real |
| Predictive Maintenance | AI4I 2020 | binary | MLP + physics features | optional real |
| HAR Activity | HAR Smartphones | seq classification | `LstmNeuralNetwork<>` | optional real |
| Air Quality | Air Quality | seq forecasting | `LstmNeuralNetwork<>` | optional real |
| Gas Sensor Drift | Gas Sensor Array Drift | 6-class + drift | MLP + batchnorm story | synthetic, UCI-inspired |

See the [Example Gallery]({{ site.baseurl }}/gallery) for the behavior plots, or
each example's own page for the full write-up.
