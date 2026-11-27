# tinymind_import_tflite (TFLite -> tinymind int8)

Importer for int8 TensorFlow Lite models, as `TFLiteConverter` writes them
with full-integer quantization. It turns a `.tflite` file into the same layer
records the ONNX importer produces, and `apps/import_onnx/tinymind_model_codegen.py`
writes two headers from them:

  * `weights.hpp`: the int8 weights, int32 biases, per-tensor activation
    grids and the requantizers;
  * `model.hpp`: one `constexpr` layer object per layer, a static memory plan
    that puts every activation in one caller-owned arena, and
    `forward(input, output, arena)`.

The generated model computes what TensorFlow Lite for Microcontrollers (TFLM)
computes, bit for bit. Every constant is derived the way TFLM's kernels
derive it: requantizer multipliers from the float scales in double, the clamps
of fused activations, MEAN's multiplier with 1/area folded in, and LOGISTIC
and TANH evaluated in TFLM's fixed-point arithmetic into 256-entry tables. The
tests compare the generated program with TFLM's own interpreter.

Reading a model needs only numpy: `tflite_flatbuffer.py` decodes the
flatbuffer without TensorFlow.

## Usage

```bash
python3 tinymind_import_tflite.py model.tflite --namespace mymodel --out-dir gen/
```

or from Python:

```python
import tinymind_import_tflite as tt
layers, generated = tt.import_tflite_model_with_code(
    "model.tflite", "gen/weights.hpp", "gen/model.hpp", "mymodel")
print(generated.plan.arena_size)
```

Then, in C++ (built with `TINYMIND_ENABLE_QUANTIZATION=1`; `FLOAT=0` and
`STD=0` work):

```cpp
#include "model.hpp"
alignas(mymodel::kModelArenaAlignment) int8_t arena[mymodel::kModelArenaSize];
mymodel::forward(input, output, arena);   // int8 NHWC in, int8 out
```

A model with a float input and output (QUANTIZE first, DEQUANTIZE last) is
imported from the int8 tensors behind them: quantize the input with
`kModelInputScale` / `kModelInputZeroPoint` and read the output on
`kModelOutputScale` / `kModelOutputZeroPoint`.

## Layout

```
apps/import_tflite/
    README.md
    Makefile                  # make test PY=<venv python>
    test_import_tflite.py     # tests against TensorFlow's schema reader and TFLM
    tflite_flatbuffer.py      # .tflite flatbuffer reader (numpy only)
    tinymind_import_tflite.py # model -> records -> weights.hpp + model.hpp
```

## Supported operators

| TFLite | tinymind |
|---|---|
| `CONV_2D` (SAME or VALID, strides; fused NONE / RELU / RELU6 / RELU_N1_TO_1) | `QConv2D`, or `QConv2DPerChannel` for per-channel weights |
| `DEPTHWISE_CONV_2D` (depth multiplier 1) | `QDepthwiseConv2D` |
| `FULLY_CONNECTED` | `QDense`, or `QConv2DPerChannel` with a 1x1 kernel for per-channel weights |
| `ADD` (same shapes) | `QAdd` |
| `MAX_POOL_2D`, `AVERAGE_POOL_2D` (no padding) | `QMaxPool2D`, `QAvgPool2D` / `QGlobalAvgPool2D` |
| `MEAN` over H and W (Keras' `GlobalAveragePooling2D`) | a window sum through one `Requantizer`, as TFLM computes it |
| `SOFTMAX` (the model's output) | `QSoftmaxFixedPoint1D` |
| `LOGISTIC`, `TANH` | 256-entry tables of TFLM's fixed-point results |
| `RELU` (same grid in and out) | `qrelu` |
| `RESHAPE`, `SQUEEZE` | no layer |
| `QUANTIZE` / `DEQUANTIZE` at the model's input / output | dropped (see above) |
| `SHAPE` / `STRIDED_SLICE` / `PACK` feeding a `RESHAPE`'s shape | ignored (Keras' `Flatten` computes its shape at run time) |

TFLite flattens in NHWC order, as tinymind does, so a dense layer after a
convolution needs no column permutation.

Anything else raises `ValueError` naming the operator and the reason; a layer
is never skipped. Refused: int16 activations (TFLite's 16x8 models), uint8
tensors, a weight zero point other than 0, dilated convolutions (the converter
writes `SPACE_TO_BATCH_ND` around them), depth multipliers above 1, padded
pooling, a broadcasting `ADD`, a softmax that is not the model's output, a
`QUANTIZE` between two layers, models with more than one subgraph, input or
output, and every operator not in the table.

`int4_dense` / `int4_conv` (`--int4-dense`, `--int4-conv`) requantize those
layers' weights to packed int4 as for an ONNX model; the result then no
longer matches TFLM bit for bit. `activation_bits=16` is not available for a
TFLite model.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install numpy tensorflow tflite-micro
make test PY=.venv/bin/python
```

  * The flatbuffer reader returns what TensorFlow's generated schema reader
    returns, field by field, for every tensor and operator.
  * The LOGISTIC and TANH tables equal TFLM's output for all 256 inputs at 7
    calibrations each, and the MEAN requantizer equals TFLM's output on six
    shapes at two calibrations each.
  * Keras models covering every operator above are converted, imported,
    generated, compiled freestanding with `-Werror` and run on random int8
    inputs (48 per model, half uniform and half on the calibration's spread);
    every output must equal TFLM's. Their weights and biases are
    randomized first, because the converter drops an untrained layer's zero
    bias. A hand-narrowed clamp checks that the fused-activation clamp is
    emitted.
  * When the MLPerf Tiny reference models are present (`make mlperf-models`
    in `benchmarks/framework_comparison_f411`, or a directory in
    `TINYMIND_MLPERF_TFLITE_DIR`), the keyword-spotting DS-CNN and ResNet-8
    are imported with no hand-written code and must match TFLM on every
    output. They do: the DS-CNN in a 16,000-byte arena (72,076 bytes with one
    buffer per output), ResNet-8 in 49,152 (114,764).
  * Refusals: dilation, depth multiplier 2, padded pooling and an unsupported
    op each raise `ValueError`.

The tests were shown to fail on planted defects: a wrong depthwise weight
transpose, a swapped SAME padding split in either axis, a logistic table one
step off, and a MEAN shift off by one. One plant passed: computing the
requantizer multiplier from a float product of the scales instead of TFLM's
double product changes the result too rarely for these inputs to show it.

`examples/tflite_digits_int8` runs a model this importer generated, in CI.
