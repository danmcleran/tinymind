---
title: PyTorch → int8 XOR
parent: Examples
nav_order: 43
layout: default
---

# PyTorch → int8 XOR

An end-to-end post-training int8 quantization demo: a PyTorch XOR MLP is trained in float32, calibrated per-tensor, and emitted as a `weights.hpp`, then run through a pure-integer TinyMind forward pass.

## How it works

- Pipeline: `nn.Linear(2,4)` → `nn.ReLU` → `nn.Linear(4,1)` → `nn.Sigmoid` in PyTorch maps to `QDense<int8,int8,int32,int8>` → `qreluBuffer` → `QDense` → int8 sigmoid LUT in TinyMind. Calibration follows TFLite / CMSIS-NN convention: symmetric per-tensor weights (`zero_point=0`), asymmetric activations, int32 biases at `input_scale * weight_scale`.
- Demonstrates the int8 affine quantization path and its host-side calibration tooling — the `(scale, zero_point)` metadata emitted by the Python script is turned back into a `Requantizer` `(multiplier, shift)` pair via `tinymind::buildRequantizer`, and the sigmoid output grid is built with `buildQSigmoidLUT`. For an MCU deployment those integer triples are baked in once on the host, so the inference binary needs neither `<cmath>` nor float math.
- Pure-integer inference classifies all four XOR corners correctly (`int8 XOR accuracy: 4/4`); the committed `weights.hpp` ships an exact textbook 2-4-1 ReLU+Sigmoid solver so the demo runs without PyTorch.

## Build and run

```bash
cd examples/pytorch_quant/xor
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

Building with `TINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` so the demo can rebuild the Requantizer and sigmoid LUT from the calibration scales). To regenerate `weights.hpp` from a fresh PyTorch training + 9×9-grid calibration run, use `make regenerate-weights` (i.e. `python3 xor_quant.py`, requires torch). The deployable MCU shape is `FLOAT=0 STD=0` with the integer `(multiplier, shift, zero_point)` triples baked in.

## Output

![int8 XOR decision surface from PyTorch-trained, pure-integer TinyMind inference]({{ site.baseurl }}/assets/plots/xor_decision_surface.png)

The heatmap sweeps the pure-integer network over the `[0,1]²` input grid. The two off-diagonal corners `(0,1)` and `(1,0)` read deep red (P(XOR=1) → 1) while the matching corners `(0,0)` and `(1,1)` read deep blue (→ 0), with the white 0.5 contour tracing the learned XOR boundary — the int8 network reproduces the classic non-linearly-separable decision surface.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/pytorch_quant/xor)
