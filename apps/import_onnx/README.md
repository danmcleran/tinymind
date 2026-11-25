# tinymind_import_onnx (ONNX -> tinymind int8)

Phase 15 importer for ONNX models quantized by ONNX Runtime's
`onnxruntime.quantization.quantize_static(...)`. It reads both of ONNX
Runtime's output formats -- QDQ (the default) and QOperator -- and emits a
tinymind-format `weights.hpp` for tinymind's int8 layers:

  * int8 symmetric weights (zero point 0), per tensor or per output channel
  * int32 biases at scale `input_scale * weight_scale`
  * per-tensor `(scale, zero_point)` for every activation tensor
  * the Requantizer `(multiplier, shift)` for every weight layer, so the
    deployable build needs no float at start-up

## Layout

```
apps/import_onnx/
    README.md
    Makefile                  # make test PY=<venv python>
    test_import_onnx.py       # end-to-end tests against ONNX Runtime
    tinymind_import_onnx.py   # parser + weights.hpp emitter
    tinymind_model_codegen.py # model.hpp: layer objects, memory plan, forward()
```

## Supported ops

| ONNX (QDQ form / QOperator form) | tinymind |
|---|---|
| `Conv` / `QLinearConv` (group 1, dilation 1, explicit pads) | `QConv2D`, or `QConv2DPerChannel` for per-channel weights |
| `Conv` / `QLinearConv` with group = channels = filters (depthwise) | `QDepthwiseConv2D` (weights `[C][KH][KW]`, one requantizer per channel) |
| `MaxPool` (QDQ, or on the int8 tensor in QOperator form; no padding, `ceil_mode` 0) | `QMaxPool2D`; input and output share one grid |
| `AveragePool` / `QLinearAveragePool`, `GlobalAveragePool` / `QLinearGlobalAveragePool` (no padding, `ceil_mode` 0) | an `AveragePool` record for `QDepthwiseConv2D` (see below) |
| `Add` / `QLinearAdd` of two activations (same shape) | an `Add` record for `QAdd` with `buildQAddParams` |
| `Gemm` / `QGemm`, `MatMul` / `QLinearMatMul` | `QDense`, or (per-channel) `QConv2DPerChannel<..., 1, 1, NumInputs, 1, 1, 1, 1, NumOutputs>`, a dense layer with one requantizer per output |
| `MatMul` + bias `Add` / `QLinearMatMul` + `QLinearAdd` | folded into one dense layer with that bias |
| `Relu` / `Sigmoid` / `Tanh` / `Softmax` / `QLinearSigmoid` / `QLinearSoftmax` (last axis) | activation records with the input and output `(scale, zero_point)` the graph uses; build the int8 LUTs (`buildQSigmoidLUT`, ...) or the softmax (`buildQSoftmaxFixedPointParams`) from them |
| `Flatten`, `Reshape` to 2D, `Identity` | no layer (see the layout notes) |

Anything else raises `ValueError`; a layer is never skipped silently. The
importer also refuses a nonzero weight zero point (tinymind weights are
symmetric), uint8 tensors (quantize with `QuantType.QInt8` for both
weights and activations), grouped convolutions other than depthwise ones,
dilated convolutions, `auto_pad` `SAME_*`, padded, dilated or `ceil_mode`
pooling, a `Gemm` with `transA` / `alpha` / `beta` other than 0 / 1 / 1, an
`Add` with broadcasting or a constant operand (other than a `MatMul`'s
bias), a `Softmax` over another axis than the last, and a weight layer whose
output is not quantized.

Every layer must read the model's input or an earlier layer's output in the
`(scale, zero_point)` that layer wrote it in. A tensor that ONNX Runtime
requantizes between two layers (a `QuantizeLinear` to another grid on the
way) is refused: no record carries that step, and a network built from the
records would compute something else. Each record's `sources` lists the
layers it reads (`-1` for the model's input; an `Add` has two), which is
what a residual network needs to wire its buffers; the header's comment for
each layer names them.

An average pool becomes a depthwise convolution whose taps all weigh
`AVG_POOL_WEIGHT` (64) and whose requantizer scales the window sum by
`input_scale / (64 * area * output_scale)`. ONNX Runtime gives an average
pool's output its own grid, which `QAvgPool2D` and `QGlobalAvgPool2D` (they
keep the input's grid) cannot express; the requantizer does the average and
the change of grid in one rounding. The weight is 64 rather than 1 so the
ratio stays below 1/4: at a larger ratio the requantizer's shift is 0 or 1
and its doubling multiply rounds to half a step first. On the test model a
weight of 1 left 77% of the 2x2 pool's outputs equal to ONNX Runtime's in
QOperator form; 64 leaves 98.6%.

Layout notes:

  * Conv weights are emitted OHWI `[F][KH][KW][C]` (ONNX stores OIHW) and
    dense weights `[out][in]` (`MatMul` stores `[in][out]`).
  * tinymind tensors are NHWC; feed the network NHWC input even though the
    ONNX model takes NCHW. A dense layer fed by a flattened 4D tensor gets
    its weight columns permuted from ONNX's NCHW flattening order to NHWC.
  * ONNX Runtime fuses a `Relu` after a `Conv` or `Gemm` into that layer's
    output range (zero point at the bottom of the int8 range), so the
    requantizer's clamp applies it and no separate layer appears.
  * A folded `MatMul` + bias `Add` rounds once, where ONNX Runtime
    quantizes the `MatMul` result to its own calibrated range and then adds
    the bias, so the two can differ when an input drives the `MatMul` result
    outside that range.

For every weight layer `k<name>_...` the header carries the input and
output scale / zero point, the geometry (`InputHeight` ... `OutputWidth`
for a conv, `NumInputs` / `NumOutputs` for a dense layer), `PerChannel`,
`WeightScale` / `Multiplier` / `Shift` (or the per-channel arrays
`WeightScales` / `Multipliers` / `Shifts`), `Weights`, `HasBias` and
`Biases`.

## Whole-model code generation (`model.hpp`)

`weights.hpp` holds every constant, but a network built from it still needs
someone to choose each layer type, wire each layer to the buffers its
`sources` name, and decide where every activation lives.
`tinymind_model_codegen.py` writes that program as a second header:

```python
from tinymind_model_codegen import import_onnx_model_with_code

layers, generated = import_onnx_model_with_code(
    "my_model_int8.onnx", "weights.hpp", "model.hpp", namespace="my_model")
print(generated.plan.arena_size)   # bytes of activation RAM the model needs
```

```cpp
#include "model.hpp"

alignas(my_model::kModelArenaAlignment) static int8_t arena[my_model::kModelArenaSize];

my_model::forward(input, output, arena);   // int8 NHWC in, int8 out
```

`model.hpp` contains:

  * one `constexpr` layer object per record (`QConv2D`, `QConv2DPerChannel`,
    `QDepthwiseConv2D`, `QDense`, `QMaxPool2D`, `QAdd`,
    `QSoftmaxFixedPoint1D`) with its weights, biases and requantizers bound
    at compile time. The objects live in flash, and the compiler folds their
    fields into the kernels;
  * the Sigmoid and Tanh lookup tables and the `QAdd` and softmax parameters,
    computed in Python by mirrors of `buildQSigmoidLUT`, `buildQTanhLUT`,
    `buildQAddParams` and `buildQSoftmaxFixedPointParams`. The tests check
    each mirror against the C++ builder over 300 random scale pairs, and the
    generated model builds with `TINYMIND_ENABLE_FLOAT=0` and
    `TINYMIND_ENABLE_STD=0`;
  * a static memory plan. Every intermediate activation gets a fixed offset
    in one caller-owned arena, and two activations share bytes when no layer
    needs both at once. A layer's output is live from the layer that writes it
    to the last layer that reads it. An element-wise layer (`Relu`, `Sigmoid`,
    `Tanh`, `Add`) writes over an input that dies there. Placement is TFLite
    Micro's greedy-by-size. The header states the arena size, the size with
    one buffer per output (`kModelUnsharedSize`), and the bytes live during
    the busiest layer (`kModelPeakLiveSize`), which no plan can go below;
  * `kModelBuffers` (offset, size, live range and the layers stored in each
    buffer) and `kModelLayerNames`, for debugging and plotting;
  * `static_assert`s that each layer type's output size matches the size
    that was planned for it.

`int4_dense=True` runs every dense layer as a `QDenseInt4` (`cpp/qint4.hpp`)
over packed int4 weights, emitted into `model.hpp`, which halves those
layers' weight bytes; `int4_conv=True` does the same for every convolution
(`QConv2DInt4`, 1x1 ones included) and depthwise layer
(`QDepthwiseConv2DInt4`). Average pools keep their int8 form. The int8
weights are dequantized, given one symmetric int4 scale per output channel
(max |w| / 7 over a dense row, a filter or a depthwise channel), and rounded
to codes in -7..7. Each bias moves to the new accumulator scale, and each
requantizer to `input_scale * s4 / output_scale` (`requantize_int4`). This
is post-training requantization with no retraining, so it costs some
accuracy. On `examples/onnx_digits_int8` the model's weights drop from 3,408
to 2,128 bytes with `int4_dense` (97.0% to 96.8% accuracy) and to 1,704
bytes with both (96.7%). `nested_namespace="int4"` lets
the int4 variant share a `weights.hpp` with the int8 one, and
`kModelWeightBytes` gives the weight bytes each variant reads.

`activation_bits=16` generates the model with int16 activations and its
int8 weights, TFLite's 16x8 scheme, from the same int8 records and with no
recalibration:

  * every activation tensor gets a symmetric int16 grid over the real range
    its int8 grid covered, `[(-128 - zp) * s, (127 - zp) * s]`, so the int16
    scale is `max(|lo|, |hi|) / 32767` (`int16_grid`). The layer that writes
    the tensor clamps to that range, so a ReLU that the int8 model folded into
    a zero point of -128 still clamps at zero;
  * weight layers keep their int8 weights. Each bias `b`, held at
    `input_scale * weight_scale`, becomes the int64 `round(b * input_scale /
    s_in16)`, and each requantizer becomes a `Requantizer<int64_t, int16_t>`
    for `s_in16 * weight_scale / s_out16` (`requantize_16x8`). A ratio of
    2^7 or more, beyond TFLite's 64-bit requantizer, raises;
  * `Add` uses TFLite's int16 parameters (`left_shift` 15, zero points 0),
    `MaxPool` and `Relu` run on int16, and `Sigmoid` / `Tanh` get 513-entry
    int16 tables (`activation_lut16`, mirroring `buildQLUT16`) read by
    `qApplyLUT16`;
  * a `Softmax` (only the model's output may be one) becomes a
    `QSoftmaxInt16`, TFLite's int16 softmax, writing probabilities on the
    2^-15 grid with zero point 0 (`softmax16_params` mirrors
    `buildQSoftmaxInt16Params`); combining `activation_bits=16` with
    `int4_dense` or `int4_conv` raises.

The model's interface then takes `int16_t` buffers. `ModelActivationType`
names the activation type in every generated model, and `kModelArenaSize`
stays in bytes. `float_reference(layers, x)` runs the records in float, with
the int8 weights, real activations and each layer clipped to its int8
range: both the int8 and the 16x8 model approximate it, so it measures what
each model's activation grids cost. On `examples/onnx_digits_int8` the 16x8
logits are 126 times closer to it than the int8 logits, for twice the
activation arena.

Include `model.hpp` (and `weights.hpp`) from one translation unit: the
constants are namespace-scope `constexpr`, so every translation unit that
includes them carries its own copy in flash.

The model's input and output are caller-owned and outside the arena. The
first layers read `input` directly, and the last layer writes `output`. The
arena needs no initialization and no contents carried between calls.
Pass `in_place=False` to give every output its own region, and `alignment=`
to change the 4-byte alignment of every offset.

On `examples/onnx_digits_int8` (a residual CNN on the optical digits) the
plan needs 1,024 bytes of the 2,048 that one buffer per output would take,
which equals the busiest layer's live bytes. Compiled for a Cortex-M0+ at
`-Os`, the model and its weights are 7.3 KB of flash (6.0 KB with
`int4_dense=True`, 5.8 KB with `int4_dense` and `int4_conv`, 9.6 KB with
`activation_bits=16`), with no `.data` and no heap.

The generator refuses a `Relu` whose output grid differs from its input's
(no record carries that rescale) and a layer that reads a softmax's output
(tinymind's softmax writes the 1/256 grid whatever the graph calibrated).

## Tests

`make test PY=<python>` with a virtual environment holding `numpy`,
`torch`, `onnx` and `onnxruntime` (never the system Python). The tests
export small PyTorch models, quantize them in both formats, per tensor and
per channel, import them, compile a C++ program that builds the matching
tinymind layer chain from the emitted header, and run both runtimes on the
same int8 inputs. The outputs agree to within 1 LSB (about 97% exactly;
ONNX Runtime and tinymind round the requantization differently). The test
model has a non-square kernel with unequal strides and padding, a second
convolution, a `Flatten` into a dense layer and a bias-free `MatMul`, so a
wrong weight layout, pad order or flattening order fails it.

A residual model (convolution, max pooling, a depthwise and a pointwise
convolution added back to their input, average and global average pooling, a
dense layer, with and without a softmax) runs through a C++ program that
wires every record by its `sources` onto `QConv2D`, `QMaxPool2D`,
`QDepthwiseConv2D`, `QAdd`, `QDense` and `QSoftmaxFixedPoint1D`. In QOperator
form the outputs agree with ONNX Runtime's to within 1 LSB (about 94%
exactly). In QDQ form ONNX Runtime averages in float and rounds ties to
even, where the integer pool rounds them away from zero (one output in eight
of a 2x2 pool), and the global pool and dense layer after it carry that
difference: within 3 LSB, more than half exact. Softmax probabilities agree
to within 3/256. Further cases check the refusals: padded and `ceil_mode`
pooling, `Concat`, groupings other than depthwise, a softmax over another
axis, and a tensor requantized between two layers.

The model code generation tests build `model.hpp` for the same models,
freestanding (`FLOAT=0`, `STD=0`), and require its outputs to equal the
hand-wired program's exactly. They do this with and without in-place
layers, for a chain, for the residual model in every format, and for a model
with standalone Sigmoid and Tanh layers, which agrees with ONNX Runtime to
within 1 LSB. Under AddressSanitizer and UBSan the model runs clean in
exactly `kModelArenaSize` bytes, and an arena 4 bytes short is reported. The
planner tests need no onnx. They check hand-built graphs and 400 random
graphs, with and without in-place layers, against a simulator that tags
every arena byte with the layer that wrote it and fails when a layer reads
an input that was overwritten. Planting a planner bug (every buffer at offset
0) fails both the simulator and the compiled comparisons.

Four more cover `int4_dense` and `int4_conv`:

* Two clamp the weights so that int4 holds them exactly (|w| <= 7, with a 7
  in every output channel): the dense layers alone, and every weight layer
  of the chain and residual models, per-channel and per-tensor. The int4
  model must then equal the int8 model bit for bit, which checks the
  packing order, rows and filters that start mid-byte, the biases and the
  requantizers. Swapping the nibble order, or reversing a depthwise layer's
  packed weights, fails them.
* Two run the real requantization against ONNX Runtime: correlation 0.996,
  mean error 3 LSB, with the dense layers in int4 or every weight layer.

Five cover `activation_bits=16`:

* The chain, residual (per-channel and per-tensor) and sigmoid / tanh
  models are generated both ways and compared with `float_reference`. The
  16x8 model's largest error must stay below 5e-4 of the output range and
  its mean error must be at least 40 times smaller than the int8 model's
  (measured: 70 to 190 times). Dropping the clamp to the int8 range,
  misscaling a bias by 2%, swapping the `Add` inputs' scales or misscaling
  a LUT by 1% fails it.
* The residual model runs clean under ASan and UBSan.
* `activation_lut16` must equal `buildQSigmoidLUT16` / `buildQTanhLUT16`
  entry for entry over 63 grids (spacing the knots as TFLite does fails
  it), and `int16_grid` is checked on a ReLU-fused grid.
* A residual model ending in a softmax tracks `float_reference` 60 to 110
  times more closely at 16 bits than at 8 over four runs (mean error 1.1e-5
  to 1.6e-5 of the probability range against 0.9e-3 to 1.5e-3); its largest
  error must stay below 2e-4. Computing the softmax's parameters from the
  int8 input scale instead of the int16 one fails it.
* `int4_dense`, `int4_conv` and a width other than 8 or 16 are refused.

Not run in CI, because torch and onnxruntime are not CI dependencies.
`examples/onnx_digits_int8` commits a generated `model.hpp`, which `make
check` and the clang example build compile and run.

## Quick example

```python
from tinymind_import_onnx import import_onnx_model

layers = import_onnx_model(
    model_path="my_model_int8.onnx",
    output_path="weights.hpp",
    namespace="my_model",
    meta={"NumInputs": 224, "NumChannels": 3},
)
```

## Requirements

`onnx` Python package is imported lazily inside `parse_onnx_model`, so
the rest of the module (the emitter) is usable without it.

## TensorFlow / Keras via ONNX

TensorFlow and Keras models reach this importer through `tf2onnx` plus
the ONNX runtime's static-quantization API. Recipe:

```bash
pip install tf2onnx onnx onnxruntime

# 1. Export your TF / Keras model to ONNX.
python -m tf2onnx.convert \
    --saved-model path/to/saved_model \
    --output model.onnx \
    --opset 13
```

```python
# 2. Post-training quantize (QDQ or QOperator, int8 weights and activations).
from onnxruntime.quantization import (
    quantize_static, QuantFormat, QuantType, CalibrationDataReader,
)

class MyCalibReader(CalibrationDataReader):
    def __init__(self, dataset):
        self._it = iter([{"input": x} for x in dataset])
    def get_next(self):
        return next(self._it, None)

quantize_static(
    "model.onnx", "model_int8.onnx",
    calibration_data_reader=MyCalibReader(calib_inputs),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    per_channel=False,   # True maps to QConv2DPerChannel
)
```

```python
# 3. Emit weights.hpp.
from tinymind_import_onnx import import_onnx_model
import_onnx_model(
    model_path="model_int8.onnx",
    output_path="weights.hpp",
    namespace="my_model",
)
```

The same recipe works for any framework that ONNX targets: JAX (via
`jax2onnx`), MXNet, PaddlePaddle, etc.

## Hybrid int8 + Q-format models

The ONNX importer covers the int8 layers. If the deployable target
inserts a Q-format hidden tier between two int8 layers (see
`apps/import_pytorch/README.md` for the `QFormatDense` / `HybridBoundary`
descriptors and `examples/mixed_precision_mlp_int8_qformat/` for the
runnable C++ counterpart), parse the ONNX model with this importer
to recover the int8 layers, then chain the result through the PyTorch
importer's emitter passing the extra `boundaries` list.
