# tinymind_import (PyTorch -> TinyMind int8)

Phase 15 importer tooling. Consumes a sequenced list of layer descriptors
(plus a calibration dataset) emitted from a trained PyTorch model and
writes a TinyMind-format `weights.hpp` containing:

  * int8 symmetric per-tensor weights (zero_point = 0, qmax = 127)
  * int32 biases at scale = `input_scale * weight_scale`
  * Per-tensor `(scale, zero_point)` for every activation

The C++ side rebuilds the Requantizers at startup via
`cpp/include/qcalibration.hpp::buildRequantizer`, or bakes the integer
`(multiplier, shift, zero_point)` triples in directly for the freestanding
inference shape.

## Layout

```
apps/import_pytorch/
    README.md
    tinymind_import.py     # core module
```

`tinymind_import.py` is a single-file Python module with no PyTorch
dependency at import time -- the caller does the PyTorch training, then
passes plain numpy weights / biases plus a numpy forward callable. This
keeps the module testable in any Python env.

## Quick example

```python
import numpy as np
from tinymind_import import (
    Dense, ReLU, Sigmoid, MinMaxObserver, PercentileObserver,
    import_pytorch_model,
)

# w1, b1, w2, b2 = numpy arrays pulled from torch.state_dict()
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
    layers, input_name="input",
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

## Layer fusion

If a `Conv2D` is immediately followed by a `BatchNorm2D` that names the
Conv2D's output as its `input_name`, the importer folds the BN constants
into the conv weights/bias before calibration. This matches the
`foldBatchNorm` math in `cpp/include/qcalibration.hpp` so the fused float
intermediate is what gets calibrated -- no late surprises.

## Observers

  * `MinMaxObserver`        -- naive min/max (= cpp `RangeObserver`)
  * `PercentileObserver`    -- e.g. 0.05 / 99.95 percentile clipping
  * `KLDivergenceObserver`  -- TensorRT-style entropy calibration

Match each layer's output statistics: heavy-tail activations want
percentile or KL; bounded post-ReLU/post-sigmoid tensors do fine with
MinMax.

## End-to-end example

`examples/import_demo/` exercises this importer on a small MLP and
verifies the C++ int8 forward against the float reference.

## Hybrid int8 + Q-format models

The importer also handles models that mix an int8 affine tier with the
TinyMind Q-format (`QValue`) pipeline -- useful when an existing
`NeuralNet<Q8.8>` hand-tuned for the MCU sits between two int8 layers
exported from PyTorch, or when a specific hidden layer wants Q-format's
compile-time fixed/fractional bit split.

Two extra descriptor kinds:

  * `QFormatDense` -- Q-format dense layer carrying float weights /
    bias plus the QValue tag (`fixed_bits`, `fractional_bits`, `signed`).
    The emitter writes raw QValue integers, no scale or zero_point.
  * `HybridBoundary` -- precision-tier transition between two adjacent
    layers (`kind = "affine_to_qvalue"` or `"qvalue_to_affine"`,
    plus a `qformat` pointer carrying the fractional-bit count).

Pass a `boundaries` list to `import_pytorch_model`; the emitter writes
one precomputed integer triple per boundary -- the same
`(multiplier, shift, zero_point)` that `cpp/qbridge.hpp::affineToQValueInt`
and `qValueToAffineInt` consume pure-integer. The deployable target shape
`TINYMIND_ENABLE_QUANTIZATION=1, FLOAT=0, STD=0` reads them as data,
no host-side helper call at startup.

```python
mid = QFormatDense(name="qfmt_mid",
                   weight=w_mid, bias=b_mid,
                   input_name="hidden",
                   forward=lambda x: x @ w_mid.T + b_mid,
                   fractional_bits=8, fixed_bits=8, signed=True,
                   observer=MinMaxObserver())
layers = [
    Dense(name="fc1", weight=w1, bias=b1, input_name="input",
          forward=lambda x: x @ w1.T + b1,
          observer=MinMaxObserver()),
    ReLU(name="hidden", input_name="fc1"),
    mid,
    Dense(name="fc2", weight=w2, bias=b2, input_name="qfmt_mid",
          forward=lambda x: x @ w2.T + b2,
          observer=MinMaxObserver()),
]
boundaries = [
    HybridBoundary(from_name="hidden", to_name="qfmt_mid",
                   kind="affine_to_qvalue", qformat=mid),
    HybridBoundary(from_name="qfmt_mid", to_name="fc2",
                   kind="qvalue_to_affine", qformat=mid,
                   qmin=-128, qmax=127),
]
import_pytorch_model(layers, ..., boundaries=boundaries)
```

`examples/mixed_precision_mlp_int8_qformat/` is the runnable C++
counterpart -- it builds the same pipeline shape with hand-crafted
weights and reports max-abs error vs the float reference.

## Importing from TensorFlow / Keras

The PyTorch importer also covers Keras / TensorFlow models via the
ONNX QDQ recipe described in `apps/import_onnx/README.md`. The short
form:

1.  Train + export TF / Keras model.
2.  Convert to ONNX: `python -m tf2onnx.convert --saved-model ... --output model.onnx`.
3.  Post-training quantize: `onnxruntime.quantization.quantize_static(
    model.onnx, model_int8.onnx, calibration_data_reader=...,
    quant_format=QuantFormat.QDQ, weight_type=QInt8, activation_type=QInt8)`.
4.  Parse with `apps/import_onnx/tinymind_import_onnx.py` and emit
    `weights.hpp`.

The hybrid int8 + Q-format flow above plugs into either entry point --
the ONNX path emits the int8 layers' descriptors, then the caller
inserts `QFormatDense` + `HybridBoundary` entries in the layer list
before calling the emitter.
