# tinymind_import_onnx (ONNX -> TinyMind int8)

Phase 15 importer for QDQ-format quantized ONNX models. Consumes the
output of `onnxruntime.quantization.quantize_static(...)` and emits a
TinyMind-format `weights.hpp` matching the layout produced by
`apps/import_pytorch/tinymind_import.py`:

  * int8 symmetric per-tensor weights (zero_point = 0)
  * int32 biases at scale = `input_scale * weight_scale`
  * Per-tensor `(scale, zero_point)` for every activation tensor

## Layout

```
apps/import_onnx/
    README.md
    tinymind_import_onnx.py
```

## Supported ops

Targets canonical CNN classifier topology:

  * `QLinearConv`         -> `QConv2D`        (per-tensor weights)
  * `QLinearMatMul`       -> `QDense`
  * `Relu` / `Sigmoid` / `Tanh` / `Softmax`   -> activation-only layers,
    emitted with TFLite-fixed output scales where applicable (Sigmoid
    1/256, Tanh 1/128, Softmax 1/256).

Heavier coverage (transformer, LSTM, attention) lives on the PyTorch
side under `apps/import_pytorch/`.

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
# 2. Post-training quantize to QDQ format.
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
    per_channel=False,
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
