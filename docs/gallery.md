---
title: Example Gallery
parent: Examples
layout: default
nav_order: 1
---

# Example Gallery

Every runnable example in [`examples/`](https://github.com/danmcleran/tinymind/tree/master/examples)
writes a header-row CSV to its `output/` directory and ships a `plot.py` that
renders the network's behavior. Reproduce any graph below with:

```bash
cd examples/<name> && make run && make plot   # writes output/*.csv and a PNG
```

The plot scripts share one style module,
[`examples/plotting/tinymind_plot.py`](https://github.com/danmcleran/tinymind/blob/master/examples/plotting/tinymind_plot.py)
(matplotlib only, headless-safe). The CSV-first contract means you can also drop
the data into pandas / a spreadsheet and build your own visualizations.

The graphs below use the dark theme to match this site; `make plot` defaults to a
light theme (readable on any viewer). Set `TINYMIND_PLOT_THEME=dark` to reproduce
these exactly.

## Training dynamics

**XOR learning curve** — Q16.16 fixed-point MLP (2→4→1).
![XOR learning curve]({{ site.baseurl }}/assets/plots/xor_learning_curve.png)

**KAN XOR learning curve** — Kolmogorov-Arnold network with learnable B-spline edges.
![KAN XOR learning curve]({{ site.baseurl }}/assets/plots/kan_xor_learning_curve.png)

## Liquid neural networks (continuous-time)

**LTC** — fused ODE-solver cell trained to a leaky-integrator step response via reverse-mode autodiff.
![LTC behavior]({{ site.baseurl }}/assets/plots/ltc_behavior.png)

**CfC** — closed-form continuous-time cell on an irregularly-sampled target (per-step `ts` into the time-gate).
![CfC behavior]({{ site.baseurl }}/assets/plots/cfc_behavior.png)

**int8 QCfC** — pure-integer CfC cell tracking the float reference, with per-step quantization error.
![int8 QCfC parity]({{ site.baseurl }}/assets/plots/qcfc_int8_parity.png)

## Physics-Informed NN

**1-D heat equation** — exact-autodiff residual training + learned field vs the analytic solution.
![PINN heat equation]({{ site.baseurl }}/assets/plots/pinn_heat1d.png)

## int8 quantization parity

**Transformer encoder block (int8)** — int8 vs float output overlay + per-element quantization error.
![Transformer int8 parity]({{ site.baseurl }}/assets/plots/transformer_int8_parity.png)

**Transformer encoder stack (int8)** — token embedding + sinusoidal positional encoding + 2 stacked linear-attention blocks, end-to-end.
![Transformer encoder stack int8 parity]({{ site.baseurl }}/assets/plots/transformer_encoder_stack_int8.png)

**Transformer encoder stack, softmax attention (int8)** — same stack with standard softmax self-attention (score grid + exp LUT).
![Transformer encoder stack softmax int8 parity]({{ site.baseurl }}/assets/plots/transformer_encoder_stack_softmax_int8.png)

**MobileNetV2-shaped pipeline (int8)** — logit parity vs the float reference.
![MobileNetV2 int8 parity]({{ site.baseurl }}/assets/plots/mobilenetv2_int8_parity.png)

## Mixture-of-Experts

**int8 MoE regime routing** — a top-1 router partitions the input domain into three regimes; one of three linear experts runs per inference (color blocks = the routing map).
![int8 MoE regime routing]({{ site.baseurl }}/assets/plots/moe_regimes_int8.png)

## UCI dataset solutions

End-to-end examples on real (or documented-synthetic) UCI datasets — see the
[UCI Dataset Capability Report]({{ site.baseurl }}/uci_dataset_capability_report)
for which example uses which dataset.

**Iris** — Q16.16 MLP (4→8→3) species classifier; training loss, test confusion, petal-space predictions.
![Iris classifier]({{ site.baseurl }}/assets/plots/iris_classifier.png)

**Energy Efficiency** — Q16.16 MLP (8→16→2) regression of building heating/cooling load; predicted vs actual.
![Energy efficiency regression]({{ site.baseurl }}/assets/plots/energy_efficiency.png)

**Optical digits** — Q16.16 MLP (64→32→10) on 8×8 handwritten-digit bitmaps; loss + 10×10 confusion.
![Optical digit classifier]({{ site.baseurl }}/assets/plots/optical_digits.png)

**Human activity recognition** — recurrent `LstmNeuralNetwork` (3→16→4) over tri-axial accelerometer windows.
![HAR activity classifier]({{ site.baseurl }}/assets/plots/har_activity.png)

**Japanese Vowels** — recurrent `ElmanNeuralNetwork` (12→16→9) speaker ID; trained offline in float, then swept across fixed-point formats. Q8.8 matches double precision at 4× smaller weights.
![Elman Japanese Vowels accuracy vs format]({{ site.baseurl }}/assets/plots/elman_vowels.png)

**Gas sensor array drift** — MLP (128→32→6) trained on batch 1; accuracy decays across later batches as sensors drift.
![Gas sensor drift]({{ site.baseurl }}/assets/plots/gas_sensor_drift.png)

**Air quality forecasting** — recurrent LSTM (1→16→1) one-step-ahead hourly pollutant forecast.
![Air quality forecast]({{ site.baseurl }}/assets/plots/air_quality_forecast.png)

## Cost & performance

**KWS pipeline per-layer cost** — compute cost + stacked weight/activation footprint per layer.
![KWS per-layer cost]({{ site.baseurl }}/assets/plots/kws_layer_cost.png)

**SIMD backends** — int8 QConv2D / QDense throughput across backends (output checksum identical — bit-exact).
![SIMD backend comparison]({{ site.baseurl }}/assets/plots/simd_backends.png)

## Applications

**XOR decision surface** — PyTorch-trained weights, pure-integer TinyMind inference; learned boundary as a heatmap.
![int8 XOR decision surface]({{ site.baseurl }}/assets/plots/xor_decision_surface.png)

**Predictive maintenance** — training loss + test confusion matrix (machine-failure classifier).
![Predictive maintenance]({{ site.baseurl }}/assets/plots/predictive_maintenance.png)

**Q-learning maze** — per-start-state navigation trajectories.
![Maze trajectories]({{ site.baseurl }}/assets/plots/maze_trajectories.png)
