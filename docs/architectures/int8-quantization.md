---
title: Int8 Affine Quantization
layout: default
parent: Architectures
nav_order: 5
---

# Int8 Affine Quantization

TinyMind ships a TFLite / CMSIS-NN style **post-training int8** layer family that lives alongside the existing `QValue` Q-format pipeline and the float pipeline. Weights and activations are int8, accumulators are int32, and a per-layer integer **Requantizer** (Q0.31 multiplier + shift) rescales between layers without any floating-point math on the inference path.

This is a different kind of "quantization" than [BinaryDense / TernaryDense]({{ site.baseurl }}/architectures/quantized-networks). Those layers binarize / ternarize the weights themselves (1- or 2-bit storage, multiply-free MACs). The int8 path keeps a full integer multiply-accumulate but maps the float distribution onto an int8 grid via a calibrated `(scale, zero_point)` per tensor.

| Path | Storage | Multiply | Calibration | When it shines |
|---|---|---|---|---|
| `BinaryDense` | 1-bit packed | XNOR + popcount | None (sign of latent weight) | Wide layers on ARMv7-M with popcount |
| `TernaryDense` | 2-bit packed | conditional add/sub | Threshold percentile | Sparse layers, multiply-free |
| **Int8 affine** (this page) | int8 weights + int8 activations | int8*int8 -> int32 MAC | Per-tensor / per-channel scale + zero_point | Drop-in TFLite-shape deployment, MobileNet-style CNN |

## Q-format vs. Affine Quantization

Both are "fixed-point", but they live in different layers of the abstraction stack:

- **`QValue` Q-format** (existing TinyMind): a compile-time bit split. `QValue<8, 8, true>` *is* an int16 with a fixed binary point. There is no runtime scale, no zero_point, no per-tensor metadata — every value in the program shares the same grid. You pick the resolution once at the type level and the compiler propagates it.
- **Int8 affine quantization** (this page): each tensor carries its own runtime `(scale, zero_point)` pair fit from observed min/max during calibration. `real = scale * (q - zero_point)`. Different tensors in the same model use different grids. The layer between them rescales with an integer `(multiplier, shift)` triple computed from the surrounding scales.

The two coexist; nothing in `qformat.hpp` or any single-`ValueType` layer changed when the int8 path was added.

## Feature Gate

The whole int8 path is behind one preprocessor flag in `cpp/include/tinymind_platform.hpp`:

```c
#define TINYMIND_ENABLE_QUANTIZATION 1
```

The deployable inference shape is `TINYMIND_ENABLE_QUANTIZATION=1, TINYMIND_ENABLE_FLOAT=0, TINYMIND_ENABLE_STD=0` — int8 weights and activations, int32 accumulators, integer requantization, no `<cmath>`. Calibration helpers in [`cpp/include/qcalibration.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/include/qcalibration.hpp) are additionally gated on `FLOAT && STD` so they only build host-side.

The `unit_test/embedded` regression matrix exercises this corner as the `quant_freestanding` build, alongside the four pre-existing `(FLOAT, STD)` corners.

## Affine Model

```
real = scale * (q - zero_point)
```

`scale` is a positive float chosen during calibration; `zero_point` is the integer storage value that decodes back to `0.0`. With `int8` storage, `q` lives in `[-128, 127]` and `zero_point` is also clamped into that range so the float zero is always representable on the grid.

For weights TinyMind follows the TFLite convention: **symmetric** (`zero_point = 0`), `qmax_signed = 127`, `-128` deliberately unused so a weight can be safely negated. For activations TinyMind uses **asymmetric** `(scale, zero_point)`, with the observed range "nudged" to include zero before fitting.

Once a layer's input scale `s_in`, weight scale `s_w`, and output scale `s_out` are fit, the integer requantization ratio is:

```
ratio = (s_in * s_w) / s_out
```

`quantizeMultiplier(ratio, multiplier, shift)` turns that float ratio into a Q0.31 multiplier and a shift exponent, both `int32`. At runtime, the layer multiplies its int32 accumulator by the multiplier with `saturatingRoundingDoublingHighMul`, applies `roundingDivideByPOT`, adds the destination `zero_point`, and saturates to `[qmin, qmax]`. Pure integer; the same code that ran on the host runs on a Cortex-M0 with no FPU.

## Core Types

[`cpp/qaffine.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qaffine.hpp) provides the runtime primitives:

```cpp
template<typename StorageType_>
struct QAffineTensor
{
    typedef StorageType_ StorageType;
#if TINYMIND_ENABLE_FLOAT
    float scale;          // host-only calibration metadata
#endif
    StorageType zero_point;
};

template<typename SrcAccum_, typename DstStorage_>
struct Requantizer
{
    int32_t multiplier;
    int32_t shift;
    DstStorageType zero_point;
    DstStorageType qmin;
    DstStorageType qmax;

    DstStorageType apply(SrcAccumType acc) const;
};
```

Note that `scale` is compiled out under `TINYMIND_ENABLE_FLOAT=0` — the freestanding inference binary holds only the integer `zero_point` plus the `(multiplier, shift)` from each layer's `Requantizer`.

The two helpers `saturatingRoundingDoublingHighMul` and `roundingDivideByPOT` follow the gemmlowp / TFLite reference semantics bit-for-bit; tests in `unit_test/quantization/` lock that down.

## Layer Family

All layers are templated on `<InputType, WeightType, AccumType, OutputType>`. Each carries its own `Requantizer` and accepts caller-owned weight / bias / lookup-table buffers — the same model can be built once on the host and re-used across many MCU targets.

| Header | Layer | Notes |
|---|---|---|
| [`qdense.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qdense.hpp) | `QDense` | Fully-connected; per-tensor weight scale |
| [`qconv2d.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qconv2d.hpp) | `QConv2D` | NHWC, VALID padding, per-tensor weight scale |
| [`qdepthwiseconv2d.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qdepthwiseconv2d.hpp) | `QDepthwiseConv2D` | **Per-channel** weight scale (TFLite mandates) |
| [`qpointwiseconv2d.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qpointwiseconv2d.hpp) | `QPointwiseConv2D` | 1x1 conv, per-tensor weight scale |
| [`qpool2d.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qpool2d.hpp) | `QMaxPool2D`, `QAvgPool2D`, `QGlobalAvgPool2D` | Max needs no requantizer (input/output share scale); avg rounds the integer sum |
| [`qactivations.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qactivations.hpp) | `qrelu`, `qrelu6`, `qApplyLUT` | Pointwise + 256-entry int8 LUTs for sigmoid / tanh; `clampForRelu` folds the activation into the upstream Requantizer's saturation pass |

Per-channel scales for `QDepthwiseConv2D` are mandatory in TFLite for accuracy reasons (the absolute weight magnitudes vary wildly across depthwise channels). The depthwise layer carries a `Requantizer` array of length `NumChannels`; everything else uses a single per-tensor `Requantizer`.

## A Minimal int8 Pipeline

```cpp
#include "qaffine.hpp"
#include "qdense.hpp"
#include "qactivations.hpp"

typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 2, 4> Fc1;
typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t, 4, 1> Fc2;

Fc1 fc1;
fc1.weights          = kFc1Weights;       // host-calibrated int8 row-major
fc1.biases           = kFc1Biases;        // int32, scale = s_in * s_w1
fc1.input_zero_point = kInputZeroPoint;
fc1.requantizer      = /* (multiplier, shift, hidden_zp, -128, 127) */;

Fc2 fc2;
fc2.weights          = kFc2Weights;
fc2.biases           = kFc2Biases;
fc2.input_zero_point = kHiddenZeroPoint;
fc2.requantizer      = /* (multiplier, shift, logit_zp, -128, 127) */;

int8_t  input_q[2], hidden_q[4], logit_q[1], output_q[1];
fc1.forward(input_q, hidden_q);
tinymind::qreluBuffer(hidden_q, 4, kHiddenZeroPoint);   // ReLU = clamp at zero_point
fc2.forward(hidden_q, logit_q);
output_q[0] = tinymind::qApplyLUT(logit_q[0], sigmoidLUT);
```

Every value on this path is integer. The `Requantizer` triples and the sigmoid LUT are computed once on the host (or by an offline tool) and embedded as `const` data on the MCU. See [PyTorch -> TinyMind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor) for the full end-to-end example with PyTorch training, calibration, and weight emission.

## Activation Functions

ReLU and ReLU6 are clamps in the integer domain. Two ways to apply them:

1. **Pointwise**: `qreluBuffer(buf, n, zero_point)` (and `qrelu6Buffer(..., q_six)`) loop over the buffer and clamp.
2. **Fused**: `clampForRelu(zero_point, qmin, qmax)` raises the upstream `Requantizer`'s `qmin` to `zero_point` so the saturation step that already runs at the end of the upstream layer covers the activation. No second pass over the buffer — this is what TFLite and CMSIS-NN do.

Sigmoid and tanh use **256-entry int8 lookup tables**. The host-side builders (`buildQSigmoidLUT`, `buildQTanhLUT`) walk every int8 input value, dequantize via the input's `(scale, zero_point)`, apply the float reference, and requantize into the destination tensor's grid. Runtime lookup is a single load:

```cpp
int8_t sigmoidLUT[tinymind::kQActivationLUTSize];   // 256 bytes
tinymind::buildQSigmoidLUT(input_scale, input_zp,
                           1.0f / 256.0f, -128,     // covers (0, 1) at full int8 res
                           sigmoidLUT);
int8_t y = tinymind::qApplyLUT(x, sigmoidLUT);
```

The LUTs themselves are pure data — drop them into flash on the MCU and the inference binary needs neither `<cmath>` nor float math.

## Calibration

[`cpp/include/qcalibration.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/include/qcalibration.hpp) gives you the host-side bridge from a float reference to the integer constants the deployable binary consumes:

| Helper | Purpose |
|---|---|
| `RangeObserver` | Streaming min/max accumulator. Sweep the float model over a representative dataset. |
| `computeAffineParamsAsymmetric(fmin, fmax, qmin, qmax)` | Activation calibration. Extends the range to include zero so `zero_point` lands on the grid. |
| `computeAffineParamsSymmetric(fmin, fmax, qmax_signed)` | Weight calibration. `zero_point = 0`, `qmax_signed = 127`. |
| `computePerChannelSymmetricScales(weights, num_channels, ...)` | Per-channel weight scales for depthwise. |
| `quantize<DstStorage>(x, scale, zp, qmin, qmax)`, `quantizeBuffer(...)` | Float -> int8 with `std::lround` rounding and saturation. |
| `buildRequantizer<DstStorage>(s_in, s_w, s_out, out_zp, qmin, qmax)` | Returns a `Requantizer<int32_t, DstStorage>` ready to embed. |

Typical workflow:

1. Train the model in float (PyTorch, NumPy, whatever).
2. Sweep a representative input set through the float forward pass; pipe each tensor's activations through a `RangeObserver`.
3. Call `computeAffineParamsAsymmetric` per activation tensor; `computeAffineParamsSymmetric` per weight tensor (`computePerChannelSymmetricScales` for depthwise).
4. `quantizeBuffer` weights to int8; quantize biases as `int32` at `bias_scale = input_scale * weight_scale`.
5. `buildRequantizer` for each layer; emit the resulting `(multiplier, shift, zero_point)` triple as a constant.
6. (For sigmoid / tanh) `buildQSigmoidLUT` / `buildQTanhLUT` and emit the 256-byte table as a constant.
7. The MCU binary defines `TINYMIND_ENABLE_QUANTIZATION=1, TINYMIND_ENABLE_FLOAT=0, TINYMIND_ENABLE_STD=0` and consumes the embedded constants.

The `examples/pytorch_quant/xor/` tutorial walks through the full pipeline, including the bit-identical Python `lround` shim that matches the C++ `std::lround` rounding.

## Deployment Footprint

The int8 path's flash story is roughly **4x smaller weights** plus a small fixed overhead for the `Requantizer` tables versus a `float` pipeline of the same shape. RAM scales similarly for activations (int8 vs float32). Compared against the existing Q8.8 (`QValue<8, 8, true>`) pipeline, weights are 2x smaller and activations are 2x smaller, with the bonus that the layer-to-layer rescale lets each tensor pick its own dynamic range instead of every value sharing the Q8.8 grid.

A side-by-side comparison runs in [`examples/kws_cortex_m_int8/`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m_int8) — same MobileNet-style depthwise-separable pipeline as the float `examples/kws_cortex_m/`, same CSV cycle/byte report format, every layer replaced with its `Q*` counterpart. See the [Keyword Spotting CNN (int8)]({{ site.baseurl }}/getting-started/keyword-spotting-int8) walkthrough for what the numbers look like.

## What Is Not Included

The int8 path is intentionally minimal:

- **No quantization-aware training (QAT)** — post-training only. The expectation is that you train in your favorite float framework and import.
- **No int4 or mixed precision** — int8 weights, int8 activations, int32 accumulators only.
- **No conv+bn+relu fusion pass** — each layer is standalone. ReLU can be folded into the upstream `Requantizer` via `clampForRelu`, but BatchNorm folding is not yet implemented.
- **Per-channel scales only on depthwise** — every other layer uses a per-tensor weight scale. (Per-tensor is enough for almost every other layer kind, and per-channel everywhere doubles requantizer storage with little accuracy gain.)

## Tests and Examples

- [`unit_test/quantization/`](https://github.com/danmcleran/tinymind/tree/master/unit_test/quantization) — Boost.Test suite. Covers `Requantizer` round-trip, per-tensor and per-channel calibration, `QConv2D` / `QDepthwiseConv2D` / `QPointwiseConv2D` / `QPool2D` / `QDense` against a float reference, and the sigmoid / tanh LUT builders.
- [`unit_test/embedded/`](https://github.com/danmcleran/tinymind/tree/master/unit_test/embedded) — `quant_freestanding` corner builds the smoke source with `FLOAT=0, STD=0, QUANT=1` to confirm the int8 path needs nothing from `<cmath>` or the standard library.
- [`examples/pytorch_quant/xor/`](https://github.com/danmcleran/tinymind/tree/master/examples/pytorch_quant/xor) — End-to-end PyTorch training + per-tensor calibration + `weights.hpp` emission + pure-integer C++ inference. Good entry point.
- [`examples/kws_cortex_m_int8/`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m_int8) — Full MobileNet-style depthwise-separable pipeline in int8, with a CSV cycle/byte report directly comparable to the float `kws_cortex_m`.

## See Also

- [Quantized Networks (Binary / Ternary)]({{ site.baseurl }}/architectures/quantized-networks) — extreme-quantization siblings; pick this path when you need 32x compression and a multiply-free MAC.
- [Q-Format (Fixed-Point)]({{ site.baseurl }}/q-format) — the existing `QValue` pipeline. Affine quantization composes with this; `QValue` does not.
- [PyTorch Interoperability]({{ site.baseurl }}/training/pytorch-interop) — for the float / Q-format weight import flow that predates the int8 path.
