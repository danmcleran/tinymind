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
    tinymind_import_onnx.py
```

## Supported ops

| ONNX (QDQ form / QOperator form) | tinymind |
|---|---|
| `Conv` / `QLinearConv` (group 1, dilation 1, explicit pads) | `QConv2D`, or `QConv2DPerChannel` for per-channel weights |
| `Gemm` / `QGemm`, `MatMul` / `QLinearMatMul` | `QDense`, or (per-channel) `QConv2DPerChannel<..., 1, 1, NumInputs, 1, 1, 1, 1, NumOutputs>`, a dense layer with one requantizer per output |
| `MatMul` + bias `Add` / `QLinearMatMul` + `QLinearAdd` | folded into one dense layer with that bias |
| `Relu` / `Sigmoid` / `Tanh` / `Softmax` / `QLinearSigmoid` | activation records with the input and output `(scale, zero_point)` the graph uses; build the int8 LUTs (`buildQSigmoidLUT`, ...) from them |
| `Flatten`, `Reshape` to 2D, `Identity` | no layer (see the layout notes) |

Anything else raises `ValueError`; a layer is never skipped silently. The
importer also refuses a nonzero weight zero point (tinymind weights are
symmetric), uint8 tensors (quantize with `QuantType.QInt8` for both
weights and activations), grouped or dilated convolutions, `auto_pad`
`SAME_*`, a `Gemm` with `transA` / `alpha` / `beta` other than 0 / 1 / 1,
and a weight layer whose output is not quantized.

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
wrong weight layout, pad order or flattening order fails it. Not run in CI,
because torch and onnxruntime are not CI dependencies.

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
