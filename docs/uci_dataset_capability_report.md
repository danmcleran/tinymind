---
title: UCI Dataset Capability Report
layout: default
nav_order: 11
---

# UCI Datasets → TinyMind Capability Report

**Date:** 2026-06-06
**Source survey:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)
**Scope:** Which TinyMind capabilities could build a working solution per dataset. More than one capability is listed where relevant. Each entry sketches the example: executable C++ inference/training + a Python/matplotlib script proving the model works.

> **Status:** the six "recommended first examples" at the bottom of this report are now built and runnable under `examples/` (`iris`, `energy_efficiency`, `optical_digits`, `har_activity`, `gas_sensor_drift`, `air_quality`). See the [Example Gallery]({{ site.baseurl }}/gallery) for their behavior plots.

---

## How to read this

Each dataset entry carries:

- **Stats** — instances / features / task, from the dataset's UCI page.
- **Primary capability** — the core TinyMind feature the example is built around.
- **Secondary capabilities** — additional features that apply (quantization tier, SIMD backend, calibration, autodiff, etc.).
- **Example shape** — what the runnable example does end-to-end.
- **Plot** — what the matplotlib script renders to demonstrate the model works.

TinyMind capability vocabulary used below (all from `cpp/`): `NeuralNet<>` (feed-forward + recurrent), `QValue` fixed-point, int8 quantization family (`QDense`/`QConv2D`/`QDepthwiseConv2D`/`QPointwiseConv2D`/`QPool2D`/`QAdd`/`QConcat`/...), recurrent int8 cells (`QLSTMCell`/`QGRUCell`), liquid/continuous-time cells (`LtcCell`, `CfCCell`, `QCfCCell`), attention (`selfattention1d`, `QAttention1D`, `QAttentionSoftmax1D`, `QMultiHeadLinearAttention1D`), signal front-ends (`conv1d`, `pool1d`, `fft1d`/`QFFT1D`, `conv2d`/`pool2d`, depthwise-separable blocks), normalization (`batchnorm`/`QBatchNorm`, `QLayerNorm1D`, `QSoftmax1D`), `PINN` autodiff (`pinn.hpp` forward/reverse dual), low-bit layers (`binarylayer`, `ternarylayer`), quantization tooling (`qcalibration.hpp` observers + CLE, `apps/import_pytorch`, `apps/import_onnx`), `qlearn` Q-learning/DQN, mixed-precision bridges (`qbridge.hpp`, `fp16_t`/`bf16_t`), SIMD backends, and the bench harness.

---

## 1. Tabular classification — small MLP

These are the canonical "fits in an MCU" fits: a 1–2 hidden-layer `NeuralNet<>` trained in float (host), deployed in `QValue` fixed-point or int8 affine.

### Iris
- **Stats:** 150 inst · 4 real features · 3-class classification · [link](https://archive.ics.uci.edu/dataset/53/iris)
- **Primary:** `NeuralNet<>` 4-8-3 MLP, `QValue` Q8.8 inference.
- **Secondary:** int8 path via `QDense` + `QSoftmax1D`; `qcalibration.hpp` per-tensor calibration; mirrors the existing `examples/xor` deployment workflow.
- **Example:** float train on host → Q8.8 + int8 forward → confusion over the 150 rows.
- **Plot:** 2-D petal-length/width scatter colored by predicted class with decision-region background; confusion matrix.

### Wine
- **Stats:** 178 inst · 13 features · 3-class · [link](https://archive.ics.uci.edu/dataset/109/wine)
- **Primary:** `NeuralNet<>` 13-16-3 MLP.
- **Secondary:** int8 `QDense` + `QSoftmax1D`; feature standardization folded into the input scale; `PercentileObserver` for the heavier-tailed chemical features.
- **Example:** float train → int8 deploy, report per-class accuracy float-vs-int8.
- **Plot:** PCA-2D scatter with predicted-class coloring; float-vs-int8 accuracy bar.

### Breast Cancer Wisconsin (Diagnostic)
- **Stats:** 569 inst · 30 real features · binary · [link](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Primary:** `NeuralNet<>` 30-16-1 MLP, sigmoid output.
- **Secondary:** int8 `QDense` + int8 sigmoid LUT (`buildQSigmoidLUT`); **`ternarylayer`/`binarylayer`** variant to show low-bit feasibility on a clean binary task; `KLDivergenceObserver` calibration.
- **Example:** float train → int8 + ternary deploy, ROC over held-out split.
- **Plot:** ROC curve (float vs int8 vs ternary), threshold-vs-sensitivity/specificity.

### Heart Disease
- **Stats:** 303 inst · 13 mixed (categorical/int/real) · classification · [link](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Primary:** `NeuralNet<>` MLP with one-hot categorical inputs.
- **Secondary:** int8 `QDense`; demonstrates mixed categorical+continuous feature encoding into a fixed input scale.
- **Example:** float train → int8 deploy; report accuracy + calibrated probability.
- **Plot:** reliability/calibration curve; per-feature contribution bar (gradient saliency via `pinn` dual).

### Parkinsons (voice)
- **Stats:** 197 inst · 22 real features · binary · [link](https://archive.ics.uci.edu/dataset/174/parkinsons)
- **Primary:** `NeuralNet<>` 22-16-1 MLP.
- **Secondary:** int8 deploy; `pinn`-dual input-gradient saliency to rank voice features.
- **Plot:** feature-importance bar from input gradients; ROC.

### Ionosphere / Spambase / Car Evaluation / Adult / Letter Recognition
- **Stats:** Ionosphere 351·34·binary; Spambase 4601·57·binary; Car 1728·6·categorical; Adult 48842·14·binary; Letter 20000·16·26-class.
- **Primary:** `NeuralNet<>` MLP (these scale the same template; Adult/Letter are the "larger tabular" stress cases).
- **Secondary:** int8 `QDense` + `QSoftmax1D` (multi-class Letter); SIMD `QDense` backend bench on the wider feature vectors (Spambase 57, Letter 16×many rows).
- **Plot:** confusion matrix (Letter 26×26 heatmap is the headline); accuracy float-vs-int8; for Spambase, precision/recall.

---

## 2. Tabular regression

Same `NeuralNet<>` template with a **linear output activation** (`forwardAs`, already covered in `unit_test/nn`).

### Energy Efficiency
- **Stats:** 768 inst · 8 features · regression (heating + cooling load) · [link](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
- **Primary:** `NeuralNet<>` 8-16-2 MLP, linear readout (two targets).
- **Secondary:** `QValue` Q16.16 inference (regression wants the wider fixed-point range); int8 with care on output scale.
- **Plot:** predicted-vs-actual scatter with y=x line for both loads; residual histogram.

### Combined Cycle Power Plant
- **Stats:** 9568 inst · 4 features · regression · [link](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)
- **Primary:** `NeuralNet<>` 4-8-1 MLP linear readout.
- **Secondary:** Q16.16; clean low-dimensional regression → good fixed-point parity demo.
- **Plot:** predicted-vs-actual scatter; error-vs-ambient-temperature.

### Concrete Compressive Strength / Airfoil Self-Noise / Auto MPG / Wine Quality / Abalone
- **Stats:** Concrete 1030·8; Airfoil 1503·5; Auto MPG 398·7; Wine Quality 4898·11; Abalone 4177·8.
- **Primary:** `NeuralNet<>` MLP linear readout (regression); Wine Quality + Abalone double as ordinal classification.
- **Secondary:** Q16.16 inference; these are the canonical nonlinear-regression benchmarks → strong float-vs-fixed-point parity story.
- **Plot:** predicted-vs-actual scatter + y=x; residual histogram; learning curve.

### Parkinsons Telemonitoring
- **Stats:** 5875 inst · 19 features · regression (UPDRS) · [link](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
- **Primary:** `NeuralNet<>` regression.
- **Secondary:** since subjects are tracked over time, a **recurrent** `NeuralNet<>` or `LtcCell` variant models progression trajectories.
- **Plot:** per-subject UPDRS trajectory predicted vs actual over time.

---

## 3. Time-series / sequential / sensor streams — recurrent + signal front-ends

This is where TinyMind's deeper bench lives: recurrent cells, conv1d/pool1d, FFT, attention, liquid cells.

### Human Activity Recognition (HAR) Using Smartphones
- **Stats:** 10299 windows · 561 engineered (or raw 9-channel inertial) · 6-class · [link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Primary:** **`conv1d` + `pool1d`** front-end on raw 9-axis inertial windows → `QDense` classifier (1-D CNN, the standard HAR-on-MCU recipe).
- **Secondary:** **`QLSTMCell`/`QGRUCell`** recurrent variant over the window; **int8 full pipeline** (`QConv1D`-style via conv stack → `QGlobalAvgPool` → `QDense` → `QSoftmax1D`); SIMD backend bench; depthwise-separable conv for the MobileNet-style efficient variant.
- **Example:** two side-by-side models (1-D CNN vs GRU), both float-trained then int8-deployed, accuracy + cycles report via bench harness.
- **Plot:** confusion matrix (6 activities); per-window activity timeline (true vs predicted) for one subject; cycles/bytes bar CNN-vs-GRU.

### Spoken Arabic Digit
- **Stats:** 8800 utterances · 13 MFCC time-series · 10-class · [link](https://archive.ics.uci.edu/dataset/195/spoken+arabic+digit)
- **Primary:** **`QGRUCell`/`QLSTMCell`** over MFCC frames → `QDense` + `QSoftmax1D`.
- **Secondary:** **`QAttention1D`/`QAttentionSoftmax1D`** pooling over frames; `CfCCell`/`QCfCCell` for the variable-length / irregular-frame angle.
- **Plot:** MFCC heatmap of one utterance with predicted digit; confusion matrix; attention-weight overlay on frames.

### EEG Eye State
- **Stats:** 14980 samples · 14 EEG channels · binary (continuous recording) · [link](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)
- **Primary:** **`QLSTMCell`/`QGRUCell`** over a sliding window of 14-channel EEG.
- **Secondary:** **`fft1d`/`QFFT1D`** band-power front-end (EEG is frequency-band driven) → `QDense`; `LtcCell`/`CfCCell` continuous-time fit for the streaming nature.
- **Plot:** EEG trace with eye-state shaded regions, predicted vs true; FFT band-power spectrogram.

### Occupancy Detection
- **Stats:** 20560 samples · 6 features (temp/humidity/light/CO2) · binary · [link](https://archive.ics.uci.edu/dataset/357/occupancy+detection)
- **Primary:** **`QGRUCell`** over the multivariate sensor stream (light/CO2 lag matters).
- **Secondary:** plain `NeuralNet<>` MLP baseline on the instantaneous reading; int8 deploy; `LtcCell` continuous-time.
- **Plot:** time-series of CO2/light with occupancy shading, predicted vs true; precision/recall.

### Gas Sensor Array Drift
- **Stats:** 13910 · 128 features · 6-gas classification across 36 months · [link](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
- **Primary:** `NeuralNet<>` MLP classifier on the 128-D sensor vector.
- **Secondary:** **`batchnorm`/`QBatchNorm`** to fight sensor drift; **cross-layer equalization (`crossLayerEqualizeDense`)** and `KLDivergenceObserver` are the headline calibration story (drift = distribution shift = exactly what robust calibration targets); per-batch accuracy over the 10 drift batches.
- **Plot:** accuracy-vs-month line (drift curve) for float / int8 / int8+CLE; per-gas confusion.

### Appliances Energy / Air Quality / Individual Household Power
- **Stats:** Appliances 19735·28; Air Quality 9358·15; Household Power 2.07M·9.
- **Primary:** **`QLSTMCell`/`QGRUCell`** sequence regression / forecasting.
- **Secondary:** `conv1d` temporal front-end; `LtcCell`/`CfCCell` continuous-time (Air Quality/Household have irregular gaps → CfC's `ts` irregular-sampling feature shines); int8 deploy.
- **Plot:** forecast vs actual over a held-out window; multi-step-ahead error growth.

### Daily and Sports Activities / Gesture Phase Segmentation
- **Stats:** Daily Sports 9120·5625 (5 IMUs, 19 activities); Gesture 9900·50 (segmentation).
- **Primary:** **`conv1d`+`pool1d`** or **`QGRUCell`** over IMU windows.
- **Secondary:** depthwise-separable 1-D conv for the 45-channel IMU input; int8 deploy + SIMD bench; attention pooling.
- **Plot:** activity/gesture-phase timeline true vs predicted; confusion matrix.

---

## 4. Image / spectrogram — 2-D conv stack

### Optical Recognition of Handwritten Digits (8×8)
- **Stats:** 5620 · 64 (8×8 bitmap 0–16) · 10-class · [link](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- **Primary:** **`conv2d` → `pool2d` → `QDense`** small CNN; the 8×8 size is ideal for an MCU-scale int8 CNN.
- **Secondary:** **full int8 pipeline** (`QConv2D` → `qrelu` → `QMaxPool2D` → `QGlobalAvgPool2D` → `QDense` → `QSoftmax1D`); depthwise-separable variant; SIMD `QConv2D` backend bench; the closest small analog to the `kws_cortex_m_int8` exemplar but on images.
- **Plot:** grid of test digits with predicted/true labels (red on miss); 10×10 confusion matrix; cycles/bytes bar float-vs-int8.

### Pen-Based Recognition of Handwritten Digits
- **Stats:** 10992 · 16 (pen-trajectory coords) · 10-class · [link](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits)
- **Primary:** `NeuralNet<>` MLP on the 16-D resampled trajectory; or **`conv1d`** over the (x,y) stroke sequence.
- **Secondary:** int8 deploy; recurrent (`QGRUCell`) over the stroke as a sequence — a nice contrast example vs the bitmap Optical-Digits version.
- **Plot:** reconstructed pen strokes with predicted digit; confusion matrix.

---

## 5. Audio / keyword-spotting style

### ISOLET (spoken letters)
- **Stats:** 7797 · 617 features · 26-class · [link](https://archive.ics.uci.edu/dataset/54/isolet)
- **Primary:** `NeuralNet<>` MLP on the 617-D feature vector → int8 `QDense` + `QSoftmax1D`.
- **Secondary:** the int8 KWS pipeline (`examples/kws_cortex_m_int8` shape) applied to real spoken-letter features; SIMD bench on the wide input.
- **Plot:** 26×26 confusion heatmap; top-confusion pairs (B/D/E etc.) bar; cycles/bytes.

### Spoken Arabic Digit
- Covered in §3 (it's MFCC time-series). Doubles as the recurrent-audio exemplar.

> Note: UCI does not host raw KWS audio like Speech Commands. ISOLET + Spoken Arabic Digit are the closest audio fits and already exercise the MFCC/feature-vector + recurrent + attention path. The synthetic `kws_cortex_m_int8` exemplar remains the raw-spectrogram reference.

---

## 6. Physics / scientific — PINN autodiff + regression

### Airfoil Self-Noise
- **Stats:** 1503 · 5 features · regression (sound pressure) · [link](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)
- **Primary:** `NeuralNet<>` regression, linear readout.
- **Secondary:** **`pinn.hpp` forward/reverse dual** for input-sensitivity analysis (∂SPL/∂frequency, ∂SPL/∂angle) — physically interpretable gradients; Q16.16 deploy.
- **Plot:** predicted-vs-actual SPL scatter; SPL-vs-frequency curves at fixed angle (model vs data); gradient-sensitivity bars.

### Concrete Compressive Strength / Energy Efficiency / Combined Cycle Power Plant
- Covered in §2 as regression. All three additionally suit **`pinn` autodiff sensitivity** demonstrations (smooth physical response surfaces).

### PINN-native (no UCI dataset needed, for contrast)
- The existing `examples/pinn_heat1d` is the true PDE-residual story (`pinn.hpp` differentiates the network twice for the heat-equation residual). UCI scientific sets are data-fit regression, not PDE-residual — worth stating explicitly so the report doesn't overclaim PINN applicability. PINN's autodiff is reusable on UCI sets only as **sensitivity/Jacobian** tooling, not residual training.

---

## 7. Reinforcement learning — `qlearn` (no UCI fit)

UCI is supervised/clustering data; it has no MDP/environment datasets. TinyMind's `qlearn` (Q-learning + DQN, `examples/maze`, `examples/dqn_maze`) has **no natural UCI dataset**. Flagging this so the report is honest: RL stays on the synthetic maze exemplars.

---

## Capability coverage summary

| TinyMind capability | Best UCI demonstrators |
|---|---|
| `NeuralNet<>` MLP (float→Q-format) | Iris, Wine, Breast Cancer, Heart, Parkinsons |
| `NeuralNet<>` regression (linear readout) | Energy Efficiency, CCPP, Concrete, Airfoil, Auto MPG |
| int8 `QDense` + `QSoftmax1D` + calibration | Iris, Wine, Letter, ISOLET, Optical Digits |
| `conv2d`/`pool2d` + int8 CNN | Optical Recognition of Handwritten Digits (8×8) |
| `conv1d`/`pool1d` temporal front-end | HAR, Daily Sports, Gesture, Pen Digits |
| `QLSTMCell` / `QGRUCell` recurrent | HAR, Spoken Arabic Digit, EEG Eye State, Occupancy, Appliances/Air Quality |
| `LtcCell` / `CfCCell` / `QCfCCell` continuous-time | Air Quality, Household Power (irregular sampling), EEG, Parkinsons Telemonitoring |
| attention (`QAttention1D` / softmax / MHA) | Spoken Arabic Digit, Daily Sports |
| `fft1d` / `QFFT1D` spectral front-end | EEG Eye State, audio sets |
| `batchnorm`/`QBatchNorm` + CLE + KL calibration | Gas Sensor Array Drift (drift = calibration headline) |
| `binarylayer` / `ternarylayer` | Breast Cancer (clean binary task) |
| `pinn.hpp` autodiff (sensitivity/Jacobian) | Airfoil, Concrete, Energy Efficiency |
| import tooling (`apps/import_pytorch` / `_onnx`) | any — PyTorch-train → import → int8 deploy |
| SIMD backends + bench harness | wide-feature sets (Letter, ISOLET, Spambase), all CNN/conv sets |
| `qbridge` mixed-precision (int8↔fp16↔Q-format) | HAR / KWS-style hybrid frontend+head pipelines |
| `qlearn` (RL) | **no UCI fit** — stays on maze exemplars |

## Recommended first examples (highest signal, lowest effort)

1. **Iris** — smallest possible end-to-end float→Q8.8→int8, mirrors `examples/xor`. Fast win, clean plots.
2. **Optical Digits 8×8** — the int8 2-D CNN story on real image data; reuses the whole `qconv2d`/`qpool2d` stack and bench harness.
3. **HAR Smartphones** — flagship recurrent + conv1d time-series, CNN-vs-GRU int8 comparison.
4. **Gas Sensor Array Drift** — the calibration/CLE/KL-observer headline (drift is the perfect motivating problem).
5. **Air Quality** — `CfCCell`/`QCfCCell` irregular-sampling forecasting, the liquid-cell differentiator.
6. **Energy Efficiency** — clean multi-output regression + fixed-point parity.

Each ships: deterministic C++ (host-trained or pre-baked weights → fixed-point/int8 forward, CSV out) + `plot.py` using the shared `examples/plotting/tinymind_plot.py` style module, per the repo's CSV-first contract.
