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

## Per-layer hooks and running one layer at a time

`forward()` has an overload that takes a hook, an object with two members:

```cpp
struct Timing
{
    void before(std::size_t) { start = cycles(); }
    void after(std::size_t layer, const int8_t* out, std::size_t size) { took[layer] = cycles() - start; }
    uint32_t start;
    uint32_t took[mymodel::kModelLayerCount];
};
Timing timing;
mymodel::forward(input, output, arena, timing);
```

`after()` receives the layer's output, valid until a later layer reuses its
bytes; `kModelLayerNames`, `kModelLayerOutputSize`, `kModelLayerOutputScale`
and `kModelLayerOutputZeroPoint` describe it, so a hook can trace activations
as well as time layers. The plain `forward(input, output, arena)` passes
`ModelNoHook`, whose empty members compile away.

`forwardLayer(i, input, output, arena)` runs layer `i` alone; calling it for
0, 1, ..., `kModelLayerCount - 1` computes what `forward()` does, so the work
can be spread across RTOS ticks or between radio events. Leave the arena
alone between the calls, and keep the input valid until layer
`kModelInputLastReader` has run.

## Streaming

A model over a window of frames (audio, vibration, any time series) that is
rerun on every new frame recomputes every frame but the newest.
`--streaming` (`streaming=True`) generates `step()` instead of `forward()`:

```bash
python3 tinymind_import_tflite.py model.tflite --namespace mymodel --out-dir gen/
python3 tinymind_import_tflite.py model.tflite --namespace mymodel --out-dir gen/ \
    --streaming --nested-namespace stream --model-header model_stream.hpp
```

```cpp
#include "model_stream.hpp"
namespace S = mymodel::stream;
alignas(S::kModelStateAlignment) int8_t state[S::kModelStateSize];
alignas(S::kModelArenaAlignment) int8_t arena[S::kModelArenaSize];
S::reset(state);
// for every new frame of S::kModelInputSize values:
S::step(frame, output, state, arena);   // valid from frame S::kModelReceptiveFrames on
```

The time axis is the input's height, or its width when the height is 1 (as
Keras' `Conv1D` converts, with `EXPAND_DIMS`). Every layer that slides a
window of K frames along time (a convolution, a depthwise convolution, a pool,
or a mean or a dense layer over the frames that remain) must have stride 1 and
no padding along time. Such a layer then reads K frames and writes one: the
tensor it reads keeps its last K frames, in order, in the caller-owned state,
so the layer reads them as one contiguous tensor and runs the same kernel
with the same requantizer as `forward()`. Each `step()` moves every history
back one frame and appends the newest. The streamed output equals
`forward()`'s for the window that ends at that frame, bit for bit, from
frame `kModelReceptiveFrames` after `reset()` on. A layer that strides or pads
along time, a softmax over several frames, or a reshape across the time axis
is refused with `ValueError`.

The MLPerf Tiny streaming wake word model imports and streams with no
hand-written code: `forward()` needs a 1,200-byte input window and a
6,656-byte arena, while `step()` needs a 40-byte frame, a 160-byte arena and
3,960 bytes of state, and does about an eighteenth of the multiply-adds per
frame. `examples/tflite_streaming_int8` and
`benchmarks/framework_comparison_f411` (`SUITE=mlperf`) run it.

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
| `CONV_2D` (SAME or VALID, strides; fused NONE / RELU / RELU6 / RELU_N1_TO_1) | `QConv2D`, or `QConv2DPerChannel` for per-channel weights; `QPointwiseConv2D` / `QPointwiseConv2DPerChannel` for an unpadded stride-1 1x1 kernel |
| `DEPTHWISE_CONV_2D` (depth multiplier 1) | `QDepthwiseConv2D` |
| `FULLY_CONNECTED` | `QDense`, or `QConv2DPerChannel` with a 1x1 kernel for per-channel weights |
| `ADD` (same shapes) | `QAdd` |
| `MAX_POOL_2D`, `AVERAGE_POOL_2D` (no padding) | `QMaxPool2D`, `QAvgPool2D` / `QGlobalAvgPool2D` |
| `MEAN` over H and W (Keras' `GlobalAveragePooling2D`) | a window sum through one `Requantizer`, as TFLM computes it |
| `SOFTMAX` (the model's output) | `QSoftmaxFixedPoint1D` |
| `LOGISTIC`, `TANH` | 256-entry tables of TFLM's fixed-point results |
| `RELU` (same grid in and out) | `qrelu` |
| `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS` | no layer |
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
    The streaming wake word model must match as well, both as `forward()`
    and streamed one frame at a time.
  * Every model driver also runs the model one `forwardLayer()` at a time
    over fresh garbage, overwriting the input after `kModelInputLastReader`,
    and requires the same output; and it checks that the hook is called once
    per layer, in order, with the layer's output size.
  * Streaming: three Keras models are streamed frame by frame and every
    output must equal `forward()`'s over the window ending there and TFLM's:
    time along the height with a tensor read by windows of 3 and 5 frames, a
    residual add, max and average pools, a mean and a dense layer over time;
    Keras `Conv1D` layers (time along the width); and a sequence-to-sequence
    model whose output depends on only 5 of its 20 frames, checked from the
    5th frame on. Each sequence runs twice with different bytes in the state
    before `reset()`, and the outputs, warm-up included, must be the same.
  * Refusals: dilation, depth multiplier 2, padded pooling and an unsupported
    op each raise `ValueError`, and so do a stride or padding along time, a
    single-frame input and a model that starts with a dense layer when
    streaming.

The tests were shown to fail on planted defects: a wrong depthwise weight
transpose, a swapped SAME padding split in either axis, a logistic table one
step off, and a MEAN shift off by one; and, for streaming and stepping, an
input or layer history slid by the wrong amount or not at all, a wrong
window offset into a history, a receptive field one frame short, a reset
that skips a history, a history one frame short, a wrong size passed to the
hook, and a wrong last reader of the input. One plant passed: computing the
requantizer multiplier from a float product of the scales instead of TFLM's
double product changes the result too rarely for these inputs to show it.

`examples/tflite_digits_int8` runs a model this importer generated in CI, and
`examples/tflite_streaming_int8` a streaming one.
