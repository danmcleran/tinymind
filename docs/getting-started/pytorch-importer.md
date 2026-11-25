---
title: PyTorch -> tinymind int8 (importer)
layout: default
parent: Getting Started
nav_order: 7
---

# PyTorch → tinymind int8 (production importer)

This tutorial walks the **production PyTorch importer flow**: take a trained PyTorch model, pull weights from `torch.state_dict`, run per-layer calibration with any of `MinMaxObserver` / `PercentileObserver` / `KLDivergenceObserver`, optionally apply Cross-Layer Equalization to recover accuracy on imbalanced layers, and emit a tinymind-format `weights.hpp` that snaps straight into the int8 `Q*` layer family.

It's the heavier-lift counterpart to [PyTorch → tinymind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor). Same destination — pure-integer C++ inference — but instead of hand-rolling the calibration loop for one tiny network, you describe each layer once and the importer handles range estimation, Conv+BN fusion, weight quantization, and header emission.

Source: [`apps/import_pytorch/tinymind_import.py`](https://github.com/danmcleran/tinymind/tree/master/apps/import_pytorch) (importer module), [`examples/import_demo/`](https://github.com/danmcleran/tinymind/tree/master/examples/import_demo) (end-to-end example).

## When to use each path

| Path | Use when |
|---|---|
| [XOR walkthrough]({{ site.baseurl }}/getting-started/pytorch-quant-xor) | One- or two-layer network, you want to see every primitive (`RangeObserver`, `computeAffineParamsAsymmetric`, `buildRequantizer`) in plain code |
| **This page** | Real PyTorch model with Conv2D + BN + ReLU stacks, dense classifier, multiple activation types. You want per-layer observer selection and automatic Conv+BN folding |
| [ONNX importer](https://github.com/danmcleran/tinymind/tree/master/apps/import_onnx) | Source model is already quantized by `onnxruntime.quantization.quantize_static` (QDQ or QOperator format). Skip Python-side calibration entirely |

## What the importer does

`apps/import_pytorch/tinymind_import.py` is a single-file Python module with **no PyTorch dependency at import**. The caller handles PyTorch training, then hands the importer plain numpy arrays plus a numpy forward callable per layer. That keeps the module testable in any Python env and isolates the deployment surface from upstream PyTorch version churn.

The four passes:

1. **Layer description.** Caller assembles an ordered list of `Dense` / `Conv2D` / `BatchNorm2D` / `ReLU` / `Sigmoid` / `Tanh` / `Softmax` descriptors, optionally led by one `LSTM` (see [Sequence models](#sequence-models-one-nnlstm-first)). Each descriptor carries the weight / bias numpy arrays (or none for activations), the predecessor name, and a numpy `forward` callable.
2. **Layer fusion.** `fuse_layers` finds Conv2D-then-BatchNorm2D pairs and folds the BN constants into the conv weights / bias **pre-quantization**. Same math as `foldBatchNorm` in `cpp/include/qcalibration.hpp`, applied on numpy.
3. **Calibration.** `calibrate` streams the calibration dataset through each layer's forward callable and feeds the output into that layer's observer (`MinMaxObserver` / `PercentileObserver` / `KLDivergenceObserver`).
4. **Quantization + header emission.** `quantize_weights` emits symmetric int8 weights + int32 biases (`bias_scale = input_scale * weight_scale`). `emit_weights_header` writes the `weights.hpp`; the shape matches `examples/pytorch_quant/xor/weights.hpp` so the consuming C++ code is identical.

Top-level `import_pytorch_model(...)` wraps all four passes into one call.

## Quick example

```python
import numpy as np
from tinymind_import import (
    Dense, ReLU, Sigmoid, MinMaxObserver, PercentileObserver,
    import_pytorch_model,
)

# w1, b1, w2, b2 — numpy arrays pulled from a trained model's torch.state_dict()
layers = [
    Dense(name="fc1", weight=w1, bias=b1, input_name="input",
          forward=lambda x: x @ w1.T + b1,
          observer=MinMaxObserver()),
    ReLU(name="hidden", input_name="fc1",
         observer=MinMaxObserver()),
    Dense(name="fc2", weight=w2, bias=b2, input_name="hidden",
          forward=lambda x: x @ w2.T + b2,
          observer=PercentileObserver(0.05, 99.95)),
    Sigmoid(name="output", input_name="fc2"),
]

ranges = import_pytorch_model(
    layers,
    input_name="input",
    input_observer=MinMaxObserver(),
    dataset=[np.array([0, 0], np.float32),
             np.array([0, 1], np.float32),
             np.array([1, 0], np.float32),
             np.array([1, 1], np.float32)],
    output_path="weights.hpp",
    namespace="my_model",
    meta={"NumInputs": 2, "HiddenSize": 4, "NumOutputs": 1},
)
```

The emitted `weights.hpp` contains:

- Symmetric per-tensor int8 weights (`zero_point = 0`, `qmax = 127`)
- int32 biases at `scale = input_scale * weight_scale`
- Per-tensor `(scale, zero_point)` for every activation tensor

The C++ side rebuilds the `Requantizer` triples at startup via `tinymind::buildRequantizer` (host-side gate `FLOAT && STD`), or bakes the integer `(multiplier, shift, zero_point)` triples directly for the freestanding inference shape.

## Picking an observer per layer

The three observers cover different activation shapes:

| Observer | Best for |
|---|---|
| `MinMaxObserver` | Bounded post-ReLU / post-sigmoid / post-tanh tensors. Naive min/max is enough — no outliers to clip |
| `PercentileObserver(lo, hi)` | Heavy-tail activations (post-conv with large receptive field, pre-softmax logits). `(0.05, 99.95)` clips the worst ~0.1% so the int8 grid is not wasted on a handful of extreme samples |
| `KLDivergenceObserver` | When percentile clipping is too crude. TensorRT-style: fix a 2048-bin histogram width, fill it, sweep threshold T in `[128, 2048]` to minimize KL between the clipped float distribution and its int8-quantized form. Heaviest but highest fidelity |

Match the observer to each tensor's empirical shape; the importer does not try to auto-pick. The [`examples/import_demo/`](https://github.com/danmcleran/tinymind/tree/master/examples/import_demo) C++ binary exercises all three on a deterministic 3-8-4-2 MLP so the calibration math is easy to inspect side by side.

## Cross-Layer Equalization

When two consecutive layers have wildly imbalanced per-channel weight magnitudes — `Conv2D(c=64)` with one channel an order of magnitude bigger than the rest, feeding into a `Conv2D` whose corresponding input slice is much smaller — naive per-tensor quantization wastes resolution on the loud channel and quantizes everything else to noise.

Cross-Layer Equalization (Nagel et al.) rebalances each channel before quantization. For each intermediate channel `c`:

```
r1 = max |W1[c, :]|
r2 = max |W2[:, c]|
s  = sqrt(r1 / r2)
W1[c, :] /= s;  b1[c] /= s;  W2[:, c] *= s
```

The model's output is **preserved exactly** under ReLU or identity activations (both are positively homogeneous). The C++ helpers `crossLayerEqualizeDense` and `crossLayerEqualizeConv2D` in `cpp/include/qcalibration.hpp` implement this; the Python importer can call them or do the same math on numpy arrays before passing weights into `import_pytorch_model`.

## Whole-model code generation

`generate_model` runs the same fusion, calibration and quantization, then hands the model to the ONNX importer's code generator, so a PyTorch model gets the same `weights.hpp` and `model.hpp` as an ONNX one: one `constexpr` layer object per layer, a static memory plan for every intermediate activation, and `forward(input, output, arena)`.

```python
from tinymind_import import Conv2D, Dense, ReLU, Softmax, MinMaxObserver, generate_model

layers = [
    Conv2D(name="c1", weight=w1_ohwi, bias=b1, input_name="x",
           input_shape=(8, 8, 1), padding=(1, 1, 1, 1),   # (top, bottom, left, right)
           forward=lambda x: conv_nhwc(x, w1_ohwi, b1, padding=(1, 1, 1, 1))),
    ReLU(name="r1", input_name="c1"),
    Dense(name="fc", weight=w2_nhwc, bias=b2, input_name="r1",
          forward=lambda x: x.reshape(-1) @ w2_nhwc.T + b2),
    Softmax(name="probs", input_name="fc"),
]
records, generated = generate_model(
    layers, input_name="x", input_observer=MinMaxObserver(), dataset=calibration,
    weights_path="weights.hpp", model_header_path="model.hpp", namespace="net")
```

It takes `Dense`, `Conv2D` (with a `BatchNorm2D` folded in), `ReLU`, `Sigmoid`, `Tanh` and a final `Softmax`; `MoE`, `LSTM` and `QFormatDense` raise `ValueError`. A `Conv2D` needs `input_shape` (its NHWC input) and takes `stride` and `padding`. A `ReLU` that is the only reader of a `Dense` or `Conv2D` output is folded into that layer's requantizer clamp. Activations are NHWC throughout, so a `Dense` after a convolution needs its weights permuted from PyTorch's NCHW flattening. The code generator's options pass through: `activation_bits=16` for 16x8, `int4_dense` and `int4_conv` for packed int4 weights.

## End-to-end exemplar — `examples/import_demo/`

The directory has two complementary entry points:

- **`import_demo.cpp`** — host binary built by `make`. Carries a deterministic 3-8-4-2 MLP with hardcoded float weights, drives a 64-sample synthetic calibration set through all three observers plus Cross-Layer Equalization, then runs both the float reference and the pure-integer int8 forward and reports max-abs error. Stands alone; **no Python or PyTorch needed**. Tolerance 0.08; the bundled seed lands around 0.004 max-abs.
- **`demo.py`** — production importer flow. Trains the same shape MLP in PyTorch, pulls numpy weights from `torch.state_dict`, then drives `apps/import_pytorch/tinymind_import.import_pytorch_model` to emit a real `weights.hpp`. Run via `make regenerate-pytorch` (requires `torch` + `numpy` via pyenv).

```sh
cd examples/import_demo
make clean && make && make run
```

Expected output:

```
CLE float drift (ReLU model): 0.0...e-05
ranges:
  input  scale=...  zp=...
  h1     scale=...  zp=...
  h2     scale=...  zp=...  (percentile [...])
  logit  scale=...  zp=...  (kl threshold=...)

import_demo parity test (16 samples)
  max |y_int8 - y_float| = 0.0...
  tolerance 0.08 : PASS
```

Exit status 0 on PASS, 1 on FAIL.

## Sequence models: one nn.LSTM first

A model that starts with one `nn.LSTM` and continues with the layers above on its last hidden state imports onto [`QLSTMCell`]({{ site.baseurl }}/architectures/lstm-gru#int8-quantized-counterparts) and the usual int8 head. Build the LSTM descriptor from the module's state dict; every calibration sample is then a whole sequence, `[step][feature]`:

```python
state = {k: v.detach().numpy() for k, v in model.lstm.state_dict().items()}
w, b = model.fc.weight.detach().numpy(), model.fc.bias.detach().numpy()
layers = [
    LSTM.from_state_dict("lstm", state, input_name="input",
                         sequence_length=24, cell_bits=16),
    Dense(name="head", weight=w, bias=b, input_name="lstm",
          forward=lambda h: h @ w.T + b),
]
import_pytorch_model(layers, input_name="input",
                     input_observer=MinMaxObserver(),
                     dataset=calibration_sequences,
                     output_path="weights.hpp", namespace="my_model")
```

The importer runs the float LSTM over every calibration sequence from zero state and observes every hidden and cell state, giving both symmetric scales (zero point 0). The cell state is int16 by default (`cell_bits=8` for int8). Weights are symmetric int8 per gate, the input and recurrent weights each with their own scale, in PyTorch's gate order (i, f, g, o), which is `QLSTMCell`'s. The gate pre-activations use a fixed 8/127 grid, and `bias_ih + bias_hh` is quantized in it as `quantizeQLSTMBiases` does. On the C++ side, fill `QLSTMScales` from the emitted floats, call `buildQLSTMParams`, and build the sigmoid and tanh LUTs on `k<name>_LutInputScale`.

Stacked, bidirectional and projected LSTMs, and an LSTM after another layer, are refused with a `ValueError`. `nn.GRU` is not supported: PyTorch applies the reset gate after the recurrent product and `QGRUCell` before it.

[`examples/import_lstm_demo/`](https://github.com/danmcleran/tinymind/tree/master/examples/import_lstm_demo) runs the whole path for a 3-input, 8-unit LSTM with a linear head over 24-step sequences. Its maximum output error against the float model is 0.063 with the int16 cell and 0.074 with the int8 cell.

## ONNX path

`apps/import_onnx/tinymind_import_onnx.py` is the ONNX counterpart. It reads a model quantized by `onnxruntime.quantization.quantize_static` in either of ONNX Runtime's formats (QDQ, the default, or QOperator), maps `Conv` / `Gemm` / `MatMul` (plus a bias `Add`) onto `QConv2D` / `QConv2DPerChannel` / `QDense`, depthwise `Conv` onto `QDepthwiseConv2D`, `MaxPool` onto `QMaxPool2D`, average pooling onto a `QDepthwiseConv2D` with equal taps, and a residual `Add` onto `QAdd` -- converting ONNX's OIHW conv weights to OHWI, `MatMul`'s `[in][out]` weights to `[out][in]`, and a flattened NCHW input to NHWC column order -- and emits a `weights.hpp` with per-tensor activation `(scale, zero_point)`, int8 weights, int32 biases and precomputed requantizers. Its code generator also writes a whole `model.hpp` (layer objects, a static activation memory plan and `forward()`) for an int8, 16x8 or packed-int4-weight model; see the [generated digits model]({{ site.baseurl }}/examples/onnx_digits_int8). Each record lists the layers it reads, and a tensor that ONNX Runtime requantizes between two layers is refused, since no record could carry that step. An op or setting it cannot map exactly (padded pooling, grouped convolution other than depthwise, `Concat`, uint8 tensors, a nonzero weight zero point) raises `ValueError` instead of being skipped. Its tests (`make test` in `apps/import_onnx` with a torch + onnxruntime venv) compile the emitted header against tinymind's layers and match ONNX Runtime's int8 outputs to within 1 LSB (3 LSB for a residual model with average pooling in QDQ form, where ONNX Runtime averages in float and rounds ties to even). The `onnx` Python package is imported **lazily** inside `parse_onnx_model` so the emitter half of the module is usable without it.

This is the right path when the source-of-truth model has already been quantized upstream and you just need to land it on tinymind.

## What this does not cover

- **QAT** — this importer is post-training only: train in float, calibrate, deploy. Quantization-aware training exists for the ONNX path's int4 weight options (`apps/import_onnx/tinymind_qat.py`).
- **Per-channel everywhere** — the importer emits per-tensor weight scales for everything except `QDepthwiseConv2D` (TFLite mandate). Use `QConv2DPerChannel` if a particular conv needs per-channel; the importer can be extended to drive it.
- **Auto observer selection** — the caller picks the observer per layer.

## See Also

- [Int8 Affine Quantization]({{ site.baseurl }}/architectures/int8-quantization) — the `Q*` layer family the emitted `weights.hpp` consumes.
- [PyTorch -> tinymind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor) — hand-rolled version of the same flow for a single tiny network.
- [`apps/import_pytorch/`](https://github.com/danmcleran/tinymind/tree/master/apps/import_pytorch) — module source.
- [`apps/import_onnx/`](https://github.com/danmcleran/tinymind/tree/master/apps/import_onnx) — ONNX importer (QDQ and QOperator formats).
- [`examples/import_demo/`](https://github.com/danmcleran/tinymind/tree/master/examples/import_demo) — exemplar.
