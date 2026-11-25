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
int4 weights accuracy:          0.9672 (same class as int8 on 1759 of 1797)
weights:                        3408 bytes int8, 1704 bytes in int4
16x8 accuracy:                  0.9688 (arena 2048 bytes, same weights)
same class as the float model:  int8 1791, 16x8 1795 of 1797
logit error vs float activations (first 256 digits): mean 0.1619 int8, 0.00128 16x8; max 0.6565 int8, 0.00633 16x8
```

![Memory plan and test confusion]({{ site.baseurl }}/assets/plots/onnx_digits_int8.png)

At the top left is the memory plan. Each box is one arena buffer, drawn across the layers its output is live for and at the bytes it occupies. The seven activations fit in 1,024 bytes, half of the 2,048 that one buffer per output would take. No plan can do better, because the two 512-byte convolution outputs are both live while the residual `Add` runs. At the top right is the confusion matrix of the generated int8 model, which matches the float model's 97.0%. The generated model predicts a different class from ONNX Runtime on only two digits. On both of them, ONNX Runtime's two highest int8 logits are one step apart.

For a Cortex-M0+ (`arm-none-eabi-g++ -std=c++14 -mcpu=cortex-m0plus -mthumb -Os -ffunction-sections -fdata-sections -c`), `model.hpp` with static buffers and a `main` that calls `forward` compiles to 7,326 bytes of code and read-only data. It has no `.data`, its 1,098 bytes of `.bss` are the arena plus the input and output buffers, and it makes no heap allocation.

`model_int4.hpp` runs every weight layer on packed per-channel int4 weights (`QConv2DInt4`, `QDepthwiseConv2DInt4`, `QDenseInt4`). That halves the weights to 1,704 bytes and brings the Cortex-M0+ footprint to 5,814 bytes, 21% less (but only 174 bytes less than with the dense layer alone in int4, since each convolution runs the int8 kernels on its unpacked weights). Accuracy goes from 97.0% to 96.7%, and the int4 variant predicts the same class as the int8 one for 1,759 of 1,797 digits. Each convolution unpacks its weights onto the stack for the length of its call and runs the int8 kernels, so the model runs at int8's speed: at `-O3` it takes 779k cycles on the Cortex-M0+ against 761k for int8 (+2.4%), 454k against 480k on the Cortex-M4 with scalar kernels, and 172k against 171k with the Arm DSP backend, where the dense layer runs on SMLAD from its packed weights. The cost is stack: the model's deepest stack grows from 636 to 1,348 bytes on the Cortex-M4 and from 1,588 to 2,092 on the Cortex-M0+ ([`benchmarks/tier1_onboard`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/tier1_onboard)).

`model_int16x8.hpp` is the same model with int16 activations and its int8 weights (TFLite's 16x8 scheme), generated from the same int8 records without recalibration: each activation gets a symmetric int16 grid over the real range its int8 grid covered, 256 times finer, and each layer clamps to that range. The bottom panel compares both models with the int8 weights run on float activations, computed for the first 256 digits by `float_reference` in the code generator. The 16x8 logits are 126 times closer on average (mean error 0.0013 against 0.162, on logits that span about 73), and the 16x8 model picks the float PyTorch model's class on 1,795 digits against 1,791. Its accuracy is 96.9%, two errors more than int8: on a classifier this small int8 activations already cost nothing, and the gain is the precision of the outputs, which a regression or feature-extraction model would keep. It costs twice the activation RAM (2,048-byte arena, 2,196 bytes of `.bss`) and 2,312 bytes more code (9,638 bytes on the Cortex-M0+); the weights are unchanged. At `-O3` it takes 854k cycles on the Cortex-M0+ against 761k for int8 (+12%), and on the Cortex-M4 453k against 480k with scalar kernels (6% less) and 203k against 171k with the Arm DSP backend (+18%).

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/onnx_digits_int8)
