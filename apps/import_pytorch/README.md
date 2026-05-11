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
