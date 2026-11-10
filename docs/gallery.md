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

**GRU XOR learning curve** — Q16.16 GRU; the four-case error falls from chance (~0.5) to ~0.06 and plateaus.
![GRU XOR learning curve]({{ site.baseurl }}/assets/plots/gru_xor.png)

**LSTM sinusoid** — Q16.16 LSTM on a held-out, half-sample-shifted sinusoid: one-step-ahead RMS error 0.0076 and auto-regressive free-run 0.011, against 0.111 for persistence.
![LSTM sinusoid prediction]({{ site.baseurl }}/assets/plots/lstm_sinusoid.png)

**LSTM sinusoid, float vs Q16.16** — the same LSTM trained in double and in Q16.16, side by side.
![LSTM sinusoid float vs Q16.16]({{ site.baseurl }}/assets/plots/lstm_sinusoid_float.png)

**Elman vs MLP on temporal XOR** — the Elman network recovers x[t] XOR x[t-1] from its recurrent memory; the MLP, with no access to x[t-1], stays near 0.5.
![Elman vs MLP on temporal XOR]({{ site.baseurl }}/assets/plots/elman_temporal_xor.png)

## Neuro-fuzzy inference

**Mackey-Glass ANFIS** — Jang's benchmark, a 16-rule Takagi-Sugeno system in 448 bytes of Q16.16: held-out fit, hybrid training, the rule base read back, and what pruning an oversized 81-rule grid recovers.
![Mackey-Glass ANFIS]({{ site.baseurl }}/assets/plots/anfis_mackey_glass.png)

**int8 ANFIS** — the same model on the int8 tier: accuracy holds, but the grade lookup tables make it 9.9x *larger* than Q16.16. A negative result, measured.
![int8 ANFIS]({{ site.baseurl }}/assets/plots/anfis_mackey_glass_int8.png)

## Liquid neural networks (continuous-time)

**LTC** — fused ODE-solver cell trained to a leaky-integrator step response via reverse-mode autodiff.
![LTC behavior]({{ site.baseurl }}/assets/plots/ltc_behavior.png)

**CfC** — closed-form continuous-time cell on irregularly sampled held-out data, against the same cell blind to the elapsed time `ts`.
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

**Encoder-decoder (seq2seq, int8)** — encoder memory + decoder with causal self-attention and cross-attention; O(1) KV-cache decode matches the full-sequence pass byte-for-byte.
![seq2seq int8 parity]({{ site.baseurl }}/assets/plots/seq2seq_int8.png)

**Autoregressive generation (int8)** — a decoder-only nano-LM generates token-by-token via `step()` over a fixed E×E KV state; int8 greedy decode reproduces the float reference's tokens (48/48).
![int8 autoregressive generation]({{ site.baseurl }}/assets/plots/tiny_generate_int8.png)

**seq2seq, softmax decoder (int8)** — the seq2seq decoder with standard softmax self/cross-attention (growing KV cache + exp LUT); incremental decode matches the full-sequence pass byte-for-byte.
![seq2seq softmax int8 parity]({{ site.baseurl }}/assets/plots/seq2seq_softmax_int8.png)

**State-space / S4-lite (int8)** — a diagonal linear-recurrent layer filters a signal with a fixed O(1) state; int8 streaming step() matches the full-sequence pass byte-for-byte.
![int8 state-space parity]({{ site.baseurl }}/assets/plots/state_space_int8.png)

**GBDT decision regions (int8)** — a gradient-boosted decision-tree ensemble classifies a 2D feature grid; int8 compare-and-branch inference reproduces the float tree's regions.
![int8 GBDT decision regions]({{ site.baseurl }}/assets/plots/gbdt_tabular_int8.png)

**MobileNetV2-shaped pipeline (int8)** — logit parity vs the float reference.
![MobileNetV2 int8 parity]({{ site.baseurl }}/assets/plots/mobilenetv2_int8_parity.png)

**ResNet residual block (int8)** — int8 vs float output across 32 elements, with per-element error.
![ResNet block int8 parity]({{ site.baseurl }}/assets/plots/resnet_block_int8.png)

**ResNet-18 stem and block (int8)** — logit parity vs the float reference; per-element error stays under 0.01.
![ResNet-18 block int8 parity]({{ site.baseurl }}/assets/plots/resnet18_block_int8.png)

**Mixed-precision KWS** — int8 frontend, fp16 attention head, int8 classifier; output parity vs float.
![Mixed-precision KWS parity]({{ site.baseurl }}/assets/plots/mixed_precision_kws.png)

**Hybrid int8 + Q-format MLP** — int8 affine and Q8.8 layers joined by pure-integer bridges; per-element error under ~0.0043.
![Hybrid int8 + Q-format MLP parity]({{ site.baseurl }}/assets/plots/mixed_precision_mlp.png)

**Importer round trip (int8)** — a 3-8-4-2 MLP calibrated with three observers and cross-layer equalization; int8 vs float parity, error near 0.004.
![Importer int8 parity]({{ site.baseurl }}/assets/plots/import_demo_parity.png)

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

**Optical digits, sample predictions** — sampled 8×8 test bitmaps titled with true and predicted labels, red when misclassified.
![Optical digits sample predictions]({{ site.baseurl }}/assets/plots/optical_digits_grid.png)

**Human activity recognition** — recurrent `LstmNeuralNetwork` (3→16→4) over tri-axial accelerometer windows; upstairs and downstairs differ only in sample order.
![HAR activity classifier]({{ site.baseurl }}/assets/plots/har_activity.png)

**Japanese Vowels** — recurrent `ElmanNeuralNetwork` (12→16→9) speaker ID; trained offline in float, then swept across fixed-point formats. Q8.8 matches double precision (92.97%) at 4× smaller weights.
![Elman Japanese Vowels accuracy vs format]({{ site.baseurl }}/assets/plots/elman_vowels.png)

**Gas sensor array drift** — MLP (128→32→6) trained on batch 1; accuracy decays across later batches as sensors drift.
![Gas sensor drift]({{ site.baseurl }}/assets/plots/gas_sensor_drift.png)

**Air quality forecasting** — recurrent LSTM (1→16→1) one-step-ahead hourly pollutant forecast.
![Air quality forecast]({{ site.baseurl }}/assets/plots/air_quality_forecast.png)

## Cost & performance

**KWS pipeline per-layer cost** — compute cost + stacked weight/activation footprint per layer.
![KWS per-layer cost]({{ site.baseurl }}/assets/plots/kws_layer_cost.png)

**KWS pipeline per-layer cost (int8)** — the int8 counterpart: per-layer compute cost and weight/activation footprint.
![KWS int8 per-layer cost]({{ site.baseurl }}/assets/plots/kws_cortex_m_int8.png)

**SIMD backends** — int8 QConv2D / QDense throughput across backends (output checksum identical — bit-exact).
![SIMD backend comparison]({{ site.baseurl }}/assets/plots/simd_backends.png)

## Applications

**XOR decision surface** — PyTorch-trained weights, pure-integer tinymind inference; learned boundary as a heatmap.
![int8 XOR decision surface]({{ site.baseurl }}/assets/plots/xor_decision_surface.png)

**XOR decision surface, Q16.16** — tinymind-trained fixed-point MLP swept over the input grid; heatmap is P(XOR=1).
![XOR Q16.16 decision surface]({{ site.baseurl }}/assets/plots/xor_q16_decision_surface.png)

**PyTorch XOR, Q16.16** — PyTorch-trained weights, fixed-point tinymind inference.
![PyTorch XOR Q16.16 decision surface]({{ site.baseurl }}/assets/plots/pytorch_xor.png)

**GRU XOR decision surface** — the trained GRU swept over the input grid, state reset at each point.
![GRU XOR decision surface]({{ site.baseurl }}/assets/plots/gru_xor_decision_surface.png)

**KAN XOR decision surface** — trained on the four XOR points only: every corner is right, but ~60 spline coefficients constrained by 4 points give a busy surface.
![KAN XOR decision surface]({{ site.baseurl }}/assets/plots/kan_xor_decision_surface.png)

**KAN XOR, dense training** — trained over the whole input region, the KAN recovers the clean four-quadrant XOR map.
![KAN XOR dense-training decision surface]({{ site.baseurl }}/assets/plots/kan_xor_dense_decision_surface.png)

**Predictive maintenance** — training loss + test confusion matrix (machine-failure classifier).
![Predictive maintenance]({{ site.baseurl }}/assets/plots/predictive_maintenance.png)

**Q-learning maze** — per-start-state navigation trajectories.
![Maze trajectories]({{ site.baseurl }}/assets/plots/maze_trajectories.png)
**DQN maze** — greedy rollouts of the trained value network from every start room.
![DQN maze trajectories]({{ site.baseurl }}/assets/plots/dqn_maze.png)

