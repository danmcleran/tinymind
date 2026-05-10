---
title: PyTorch -> TinyMind int8 (XOR)
layout: default
parent: Getting Started
nav_order: 6
---

# PyTorch -> TinyMind int8 Quantization (XOR)

This tutorial walks the smallest possible end-to-end **post-training int8 affine quantization** pipeline: train a 2-4-1 XOR MLP in PyTorch with float32, fit per-tensor activation scales over a small calibration grid, emit a `weights.hpp` with int8 weights and int32 biases, then run a pure-integer C++ forward pass through `QDense` + `qrelu` + `QDense` + an int8 sigmoid LUT.

It's the affine-quantization sibling of [`examples/pytorch/xor/`](https://github.com/danmcleran/tinymind/tree/master/examples/pytorch/xor), which uses the existing single-`ValueType` Q16.16 fixed-point path. Same problem, same network shape, different deployment lane.

Source: [`examples/pytorch_quant/xor/`](https://github.com/danmcleran/tinymind/tree/master/examples/pytorch_quant/xor).

## Why This Example?

XOR is the smallest non-trivial classification task that needs a hidden layer, and a 2-4-1 MLP fits a single page of code. That makes it a good lens on the moving parts of an int8 deployment without any layer-shape complexity getting in the way:

- A real **PyTorch training loop** in float32.
- A real **per-tensor activation calibration** step (small grid here, but the same `RangeObserver` API works on any dataset).
- Symmetric int8 weight quantization, asymmetric int8 activation quantization, and `int32` biases held at `bias_scale = input_scale * weight_scale` — the canonical TFLite / CMSIS-NN convention.
- A `Requantizer` rebuilt at startup from float scales via `tinymind::buildRequantizer`. For an MCU deployment that step happens once on the host and the resulting integer triple is embedded as a constant.
- A 256-byte int8 sigmoid LUT, also rebuilt host-side via `buildQSigmoidLUT`.

## Pipeline

```
float32 PyTorch                  int8 TinyMind
---------------                  -------------
  nn.Linear(2, 4)      -->     QDense<int8,int8,int32,int8>
  nn.ReLU              -->     qreluBuffer (or fold into upstream Requantizer)
  nn.Linear(4, 1)      -->     QDense<int8,int8,int32,int8>
  nn.Sigmoid           -->     buildQSigmoidLUT + qApplyLUT
```

## Step 1 — Train the float reference

[`xor_quant.py`](https://github.com/danmcleran/tinymind/blob/master/examples/pytorch_quant/xor/xor_quant.py) defines a textbook two-layer MLP and trains it with BCE + SGD:

```python
class XORNet(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))
```

Nothing here is TinyMind-specific — it is a vanilla PyTorch model. The whole quantization story sits **after** training is finished.

## Step 2 — Observe activation ranges

Calibration runs the trained float model over a representative input distribution and records the min / max of every intermediate tensor. For XOR the input space is just `[0, 1]^2`, so a 9 x 9 grid plus the four exact corners is enough:

```python
def calibration_grid():
    pts = np.linspace(0.0, 1.0, 9)
    return torch.from_numpy(np.array([[x, y] for x in pts for y in pts],
                                      dtype=np.float32))

with torch.no_grad():
    cal_full = torch.cat([calibration_grid(), X], dim=0)
    traced = model.trace(cal_full)        # returns input, fc1_pre, hidden, logit, output
    ranges = {name: (float(t.min()), float(t.max())) for name, t in traced.items()}
```

For larger / non-toy tasks you replace the grid with a representative slice of the deployment distribution; the rest of the emit logic is unchanged.

## Step 3 — Per-tensor calibration

The Python script reproduces TinyMind's `computeAffineParamsAsymmetric` exactly so the metadata it emits is bit-identical to what the C++ `Requantizer` expects:

```python
def affine_asymmetric(fmin, fmax, qmin=-128, qmax=127):
    if fmin > 0.0: fmin = 0.0
    if fmax < 0.0: fmax = 0.0
    qrange = float(qmax - qmin)
    if fmax == fmin or qrange == 0.0:
        return 1.0, qmin
    scale = (fmax - fmin) / qrange
    zp = lround(qmin - fmin / scale)
    return scale, max(qmin, min(qmax, int(zp)))
```

The `lround` helper is a deliberate Python shim that matches C++ `std::lround` semantics (ties round away from zero). Python's built-in `round()` uses banker's rounding (ties to even), which would silently bias the calibration constants by one ULP at edge cases. Matching `std::lround` here keeps the calibration metadata bit-identical to what the C++ `Requantizer` would compute.

Weights are calibrated symmetrically with `qmax_signed = 127` (the `-128` slot is left unused so a weight can be negated without overflow):

```python
def symmetric_weight_scale(w, qmax_signed=127):
    return float(np.max(np.abs(w))) / qmax_signed
```

Biases are held in int32 at `bias_scale = input_scale * weight_scale`, so they can be added directly to the int32 accumulator without further rescaling.

## Step 4 — Emit `weights.hpp`

The script writes a generated header with a fixed namespace and a fixed set of constants:

```cpp
namespace pytorch_quant_xor {
    constexpr float   kInputScale         = ...;
    constexpr int32_t kInputZeroPoint     = ...;
    constexpr float   kHiddenScale        = ...;
    constexpr int32_t kHiddenZeroPoint    = ...;
    constexpr float   kLogitScale         = ...;
    constexpr int32_t kLogitZeroPoint     = ...;
    constexpr float   kSigmoidOutScale    = 0.00390625f;   // 1/256
    constexpr int32_t kSigmoidOutZeroPoint= -128;          // covers (0, 1)

    constexpr float   kFc1WeightScale = ...;
    constexpr float   kFc2WeightScale = ...;

    constexpr int8_t  kFc1Weights[8]  = { ... };
    constexpr int32_t kFc1Biases[4]   = { ... };
    constexpr int8_t  kFc2Weights[4]  = { ... };
    constexpr int32_t kFc2Biases[1]   = { ... };
}
```

The committed `weights.hpp` ships an exact textbook 2-4-1 ReLU+Sigmoid solver so the example builds and runs without requiring PyTorch. Re-running `xor_quant.py` overwrites the file with whatever the SGD trainer converges to.

## Step 5 — C++ inference

[`xor_quant.cpp`](https://github.com/danmcleran/tinymind/blob/master/examples/pytorch_quant/xor/xor_quant.cpp) builds two `QDense` layers and a sigmoid LUT, then runs the four XOR inputs through pure-integer math:

```cpp
typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t,
                         pq::NumInputs, pq::HiddenSize> QDenseFC1;
typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t,
                         pq::HiddenSize, pq::NumOutputs> QDenseFC2;

QDenseFC1 fc1;
fc1.weights          = pq::kFc1Weights;
fc1.biases           = pq::kFc1Biases;
fc1.input_zero_point = static_cast<int8_t>(pq::kInputZeroPoint);
fc1.requantizer      = tinymind::buildRequantizer<int8_t>(
    pq::kInputScale, pq::kFc1WeightScale,
    pq::kHiddenScale, pq::kHiddenZeroPoint,
    -128, 127);

// fc2 is built the same way with hidden -> logit scales.

int8_t sigmoidLUT[tinymind::kQActivationLUTSize];
tinymind::buildQSigmoidLUT(pq::kLogitScale, pq::kLogitZeroPoint,
                           pq::kSigmoidOutScale, pq::kSigmoidOutZeroPoint,
                           sigmoidLUT);
```

The forward pass for one input:

```cpp
int8_t input_q[2], hidden_q[4], logit_q[1], output_q[1];

for (std::size_t k = 0; k < 2; ++k)
    input_q[k] = tinymind::quantize<int8_t>(
        kXOR[i][k], pq::kInputScale, pq::kInputZeroPoint, -128, 127);

fc1.forward(input_q, hidden_q);
tinymind::qreluBuffer(hidden_q, 4, static_cast<int8_t>(pq::kHiddenZeroPoint));
fc2.forward(hidden_q, logit_q);
output_q[0] = tinymind::qApplyLUT(logit_q[0], sigmoidLUT);
```

The only floats anywhere on this path are in the host-side `buildRequantizer` and `buildQSigmoidLUT` calls — both of which produce integer constants that an MCU deployment would embed directly, with no float math at runtime.

## Build and Run

```bash
# (Optional) regenerate weights.hpp from PyTorch training. Requires torch.
python3 xor_quant.py

# Build and run the int8 inference pipeline.
cd examples/pytorch_quant/xor
make clean && make && make run
```

Expected output:

```
int8 XOR accuracy: 4/4
```

## Porting to an MCU

The example is structured so the inference path is already MCU-ready:

1. Drop the `qcalibration.hpp` include and replace the `buildRequantizer` and `buildQSigmoidLUT` calls with the integer constants those functions emit on the host. Embed those constants in the binary.
2. Compile with `-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=0 -DTINYMIND_ENABLE_STD=0`. The forward path needs only `qaffine.hpp`, `qdense.hpp`, and `qactivations.hpp` — no `<cmath>`, no `<vector>`, no float ops.
3. The `unit_test/embedded` regression matrix builds the smoke source in exactly that corner (`quant_freestanding`) — that's the proof the inference path works at `STD=0`.

## See Also

- [Int8 Affine Quantization]({{ site.baseurl }}/architectures/int8-quantization) — what `QDense`, `Requantizer`, and the calibration helpers are doing under the hood.
- [Keyword Spotting CNN (int8)]({{ site.baseurl }}/getting-started/keyword-spotting-int8) — same int8 pipeline applied to a full MobileNet-style depthwise-separable CNN, with a CSV cycle/byte report.
- [PyTorch Interoperability]({{ site.baseurl }}/training/pytorch-interop) — the float / Q-format weight import flow that this example sits alongside.
