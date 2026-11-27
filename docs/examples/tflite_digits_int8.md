---
title: Generated TFLite Model (int8 digits)
parent: Examples
nav_order: 62
layout: default
---

# Generated TFLite Model (int8 digits)

A residual int8 CNN classifies the UCI optical digits, and none of the model code is written by hand. Keras trains it, TensorFlow's `TFLiteConverter` quantizes it to an int8 `.tflite` file, and `apps/import_tflite` generates two headers from that file: `weights.hpp` holds the constants, and `model.hpp` holds the layer objects, a static memory plan for every activation, and `forward(input, output, arena)`. Every output the generated model computes equals the one TensorFlow Lite for Microcontrollers (TFLM) computes for the same input.

## How it works

- Model: `Conv 3x3 (8), ReLU` → a residual block (`depthwise 3x3, ReLU6` → `Conv 1x1 (8)`, added back to its input) → `MaxPool 2x2` → `Conv 3x3 (16), ReLU` → `GlobalAveragePooling2D` → `Dense (10)` → softmax, on 8x8x1 input. `train.py` trains it in Keras (Adam, 60 epochs, seed 1) and converts it with full-integer int8 quantization and 512 calibration digits.
- The converter writes `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `MAX_POOL_2D`, `MEAN` (what `GlobalAveragePooling2D` becomes), `FULLY_CONNECTED` and `SOFTMAX`. `tinymind_import_tflite` reads the flatbuffer with numpy alone and turns each operator into a record, deriving every constant the way TFLM's kernels do: requantizer multipliers from the float scales in double, the clamps of fused activations, and MEAN's multiplier with 1/area folded in. The ONNX importer's code generator then writes `model.hpp` from the records.
- The memory planner treats a layer's output as live from the layer that writes it to the last layer that reads it, and places buffers largest first at the lowest offset clear of every buffer it is live beside (TFLM's greedy-by-size placement). The residual `ADD` writes its result over the first convolution's output.
- `tflite_digits_int8.cpp` reads `optdigits.tes`, maps each pixel through the input table, calls `forward()`, and compares all ten outputs of every digit with TFLM's, which `train.py` recorded in `reference.hpp`. It exits with an error if any differs. It is built with `TINYMIND_ENABLE_FLOAT=0` and `TINYMIND_ENABLE_STD=0`.

## Build and run

```bash
cd examples/tflite_digits_int8
make release
make run
make bench     # microseconds per inference on the host
make golden    # int8 outputs of the first 16 test digits
make plot      # needs matplotlib in a venv
```

The `.tflite` model, the generated headers and the TFLM reference are committed. `make regenerate PY=<venv python>` retrains and regenerates them; the venv needs `numpy`, `tensorflow` and `tflite-micro`.

## Output

```
tinymind int8 accuracy:         0.9610
TFLM int8 accuracy:             0.9610
float (Keras) accuracy:         0.9610
same 10 outputs as TFLM:        1797 of 1797 digits (0 of 17970 outputs differ)
same class as Keras:            1788 of 1797
activation arena:               1536 bytes (unshared 2460, busiest layer 1536)
weights:                        1520 bytes
```

All 17,970 outputs equal TFLM's. The activations fit in 1,536 bytes, against 2,460 with one buffer per output, and 1,536 bytes is the minimum: that many are live during the busiest layer.

![Generated TFLite model memory plan]({{ site.baseurl }}/assets/plots/tflite_digits_int8.png)

On the left, each box is an arena buffer, spanning the layers its output is live for and sitting at the bytes it occupies. On the right is the confusion matrix of the generated model on the test digits.

## Beyond this example

The importer's tests run the MLPerf Tiny reference models through it with no hand-written code: the keyword-spotting DS-CNN and ResNet-8 both match TFLM on every output, the DS-CNN in a 16,000-byte arena (72,076 bytes with one buffer per output) and ResNet-8 in 49,152 (114,764). The supported operators, the refusals and the tests are listed in [`apps/import_tflite/README.md`](https://github.com/danmcleran/tinymind/tree/master/apps/import_tflite).
