---
title: Generated ONNX Model (int8 digits)
parent: Examples
nav_order: 62
layout: default
---

# Generated ONNX Model (int8 digits)

A residual int8 CNN classifies the UCI optical digits, and none of the model code is written by hand. PyTorch trains it, ONNX Runtime quantizes it, and `apps/import_onnx` generates two headers: `weights.hpp` holds the constants, and `model.hpp` holds the layer objects, a static memory plan for every activation, and `forward(input, output, arena)`.

## How it works

- Model: `Conv 3x3 (8)` → a residual block (`Conv 3x3 (8)` added back to its input) → `MaxPool 2x2` → a depthwise-separable block (`depthwise 3x3` → `pointwise 1x1 (16)`) → flatten → `dense (10)`, on 8x8x1 input. `train.py` trains it in PyTorch (Adam, 60 epochs, seed 1) and quantizes it with `onnxruntime.quantization.quantize_static` (QDQ, per-channel weights, 512 calibration digits).
- `tinymind_import_onnx.parse_onnx_model` turns the quantized graph into records. `tinymind_model_codegen.emit_model_header` then gives each record a `constexpr` tinymind layer object (`QConv2DPerChannel`, `QAdd`, `QMaxPool2D`, `QDepthwiseConv2D`). Weights, biases and requantizers are bound at compile time, so the objects live in flash.
- The memory planner treats a layer's output as live from the layer that writes it to the last layer that reads it. Buffers are placed largest first, each at the lowest offset clear of every buffer it is live beside, which is TFLite Micro's greedy-by-size placement. The residual `Add` writes its result over the first convolution's output, because the `Add` is the last layer to read that output.
- `onnx_digits_int8.cpp` reads `optdigits.tes`, maps each pixel through the generated input table, and calls `forward()`. It is built with `TINYMIND_ENABLE_FLOAT=0` and `TINYMIND_ENABLE_STD=0`.

## Build and run

```bash
cd examples/onnx_digits_int8
make release
make run
make bench     # microseconds per inference on the host
make golden    # int8 logits of the first 16 test digits
make plot      # needs matplotlib in a venv
```

The generated headers are committed. `make regenerate PY=<venv python>` retrains the model and regenerates them; the venv needs `torch`, `onnx`, `onnxruntime` and `numpy`.

## Output

```
tinymind int8 accuracy:         0.9699
ONNX Runtime int8 accuracy:     0.9699
float (PyTorch) accuracy:       0.9699
same class as ONNX Runtime:     1795 of 1797
activation arena:               1024 bytes (unshared 2048, busiest layer 1024)
```

![Memory plan and test confusion]({{ site.baseurl }}/assets/plots/onnx_digits_int8.png)

On the left is the memory plan. Each box is one arena buffer, drawn across the layers its output is live for and at the bytes it occupies. The seven activations fit in 1,024 bytes, half of the 2,048 that one buffer per output would take. No plan can do better, because the two 512-byte convolution outputs are both live while the residual `Add` runs. On the right is the confusion matrix of the generated int8 model, which matches the float model's 97.0%. The generated model predicts a different class from ONNX Runtime on only two digits. On both of them, ONNX Runtime's two highest int8 logits are one step apart.

For a Cortex-M0+ (`arm-none-eabi-g++ -mcpu=cortex-m0plus -Os`), `model.hpp` with a `main` that calls `forward` compiles to 7,326 bytes of code and read-only data. It has no `.data`, its 1,098 bytes of `.bss` are the arena plus the input and output buffers, and it makes no heap allocation.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/onnx_digits_int8)
