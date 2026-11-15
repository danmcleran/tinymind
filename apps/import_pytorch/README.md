# tinymind_import (PyTorch -> tinymind int8)

Phase 15 importer tooling. Consumes a sequenced list of layer descriptors
(plus a calibration dataset) emitted from a trained PyTorch model and
writes a tinymind-format `weights.hpp` containing:

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
`foldBatchNorm` math in `cpp/include/qcalibration.hpp`. The fused layer's
float `forward` runs the conv and then the BN, so calibration measures the
range after the BN, which is what the fused int8 layer outputs.

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

## Sequence models: one nn.LSTM first

A model that starts with one `nn.LSTM` layer, followed by any of the layers
above on its last hidden state, maps onto `cpp/qlstm.hpp::QLSTMCell` plus the
usual int8 head. Each calibration sample is then a whole sequence,
`[step][feature]`:

```python
from tinymind_import import LSTM, Dense, MinMaxObserver, import_pytorch_model

state = {k: v.detach().numpy() for k, v in model.lstm.state_dict().items()}
w, b = (model.fc.weight.detach().numpy(), model.fc.bias.detach().numpy())
layers = [
    LSTM.from_state_dict("lstm", state, input_name="input",
                         sequence_length=24, cell_bits=16),
    Dense(name="head", weight=w, bias=b, input_name="lstm",
          forward=lambda h: h @ w.T + b),
]
import_pytorch_model(layers, input_name="input",
                     input_observer=MinMaxObserver(),
                     dataset=calibration_sequences,   # each [24][num_features]
                     output_path="weights.hpp", namespace="my_model")
```

  * The cell runs each sequence from zero state; the tensor named after the
    LSTM is its **last** hidden state, which the next layer reads.
  * `observer` sees every hidden state of every calibration step, and
    `cell_observer` every cell state. Both scales are symmetric (zero point
    0); the hidden scale is the grid the head reads.
  * `cell_bits=16` (the default) stores the cell state as int16, 256 times
    finer than `cell_bits=8`; pick 8 only when the memory matters more.
  * Weights are symmetric int8 **per gate**, input and recurrent weights
    each with their own scale. PyTorch's gate order (i, f, g, o) is
    QLSTMCell's, so the state-dict tensors are used as they are.
  * The gate pre-activations use a fixed 8/127 grid (`LSTM_LUT_INPUT_SCALE`;
    the gates saturate beyond +-8), and the int32 biases
    (`bias_ih + bias_hh`) are quantized in that grid the way
    `quantizeQLSTMBiases` does.

The emitted header carries `k<name>_CellScale`, `k<name>_LutInputScale`,
`k<name>_InputWeightScales[4]`, `k<name>_RecurrentWeightScales[4]`, the int8
`k<name>_InputWeights` / `k<name>_RecurrentWeights` (gate-major), the int32
`k<name>_Biases`, the cell range and a `<name>_CellType` typedef. The C++ side
fills `QLSTMScales` from them, calls `buildQLSTMParams`, and builds the
sigmoid and tanh LUTs on the LUT input scale; `examples/import_lstm_demo/`
does exactly that and checks the result against the float model.

`LSTM.from_state_dict` raises `ValueError` for what `QLSTMCell` cannot run: a
stacked (`num_layers > 1`), bidirectional or projected (`proj_size`) LSTM. An
LSTM anywhere but first is refused as well. `nn.GRU` has no importer path:
PyTorch applies the reset gate after the recurrent product and `QGRUCell`
applies it before, so a PyTorch GRU does not map onto the cell.

## Hybrid int8 + Q-format models

The importer also handles models that mix an int8 affine tier with the
tinymind Q-format (`QValue`) pipeline -- useful when an existing
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

## Tests

```
python3 -m venv .venv && .venv/bin/pip install numpy
cd apps/import_pytorch && ../../.venv/bin/python -m unittest test_import_pytorch
```

Conv2D + BatchNorm2D fusion: the fused weights compute conv-then-BN, and
calibration observes the range after the BN.

LSTM: the float forward against the LSTM equations written independently,
the hidden and cell scales, the per-gate weight scales and their gate-major
layout, the bias sum and its rounding, the emitted constants and the
rejections. With `torch` installed in the venv, one more test checks the
float forward against `nn.LSTM` itself (parameters scaled so the gates
saturate) and the rejections against real stacked, bidirectional and
projected modules; without torch it is skipped.

These tests are not run in CI.
