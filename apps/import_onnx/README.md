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
