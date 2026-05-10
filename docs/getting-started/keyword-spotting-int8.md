---
title: Keyword Spotting CNN (int8)
layout: default
parent: Getting Started
nav_order: 7
---

# Keyword Spotting CNN on a Cortex-M (int8)

This tutorial is the **int8 quantized counterpart** to the float [Keyword Spotting CNN]({{ site.baseurl }}/getting-started/keyword-spotting-cnn) example. Same MobileNet-style depthwise-separable pipeline, same input shape, same CSV cycle/byte report format from the bench harness, but every layer is replaced with its `Q*` sibling from `cpp/q*.hpp`. Side-by-side the two runners give you a direct apples-to-apples comparison of footprint and cycle counts between a `float` pipeline and the int8 affine pipeline.

Source: [`examples/kws_cortex_m_int8/`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m_int8).

## Pipeline

```
input [20 x 20 x 1] int8
  -> QConv2D 3x3, 8 filters           -> [18 x 18 x 8]  int8
  -> QMaxPool2D 2x2                   -> [9  x 9  x 8]  int8
  -> QDepthwiseConv2D 3x3 (per-chan)  -> [7  x 7  x 8]  int8
  -> QPointwiseConv2D 8 -> 16         -> [7  x 7  x 16] int8
  -> QGlobalAvgPool2D                 -> [16]           int8
  -> QPointwiseConv2D (dense) 16 -> 10 -> [10]           int8 logits
```

The structural parallels with the float pipeline are deliberate. Each `Q*` layer occupies the same slot in the chain as its float counterpart and its compile-time output dimensions feed the next layer's compile-time inputs the same way:

```cpp
using QConv1Type = tinymind::QConv2D<int8_t, int8_t, int32_t, int8_t,
                                     20, 20, 1, 3, 3, 1, 1, 8>;
using QPool1Type = tinymind::QMaxPool2D<int8_t,
                                        QConv1Type::OutputHeight,
                                        QConv1Type::OutputWidth,
                                        8, 2, 2, 2, 2>;
using QDwType    = tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t,
                                              QPool1Type::OutputHeight,
                                              QPool1Type::OutputWidth,
                                              8, 3, 3, 1, 1>;
// ...QPwType, QGapType, QDenseType
```

`QDepthwiseConv2D` carries a per-channel `Requantizer` array (TFLite mandate); every other layer uses a single per-tensor `Requantizer`. Max-pooling carries no `Requantizer` at all because the input and output share the same `(scale, zero_point)`.

## Calibration in This Runner

The host runner generates synthetic float weights and converts them with the helpers in [`cpp/include/qcalibration.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/include/qcalibration.hpp):

- `quantize` / `quantizeBuffer` — symmetric int8 weight quantization
- `computePerChannelSymmetricScales` — per-channel scale fit for the depthwise weights
- `buildRequantizer` — combines `(input_scale, weight_scale, output_scale, output_zero_point)` into the integer `(multiplier, shift)` pair the `Requantizer` consumes at runtime

A real deployment would feed observed activation ranges from a representative dataset through `RangeObserver` + `computeAffineParamsAsymmetric` to fit per-tensor activation scales (see the [PyTorch -> TinyMind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor) tutorial for that flow end-to-end). This runner uses fixed scales (`1/127` for activations, `0.1/127` for weights) since the demo's job is footprint and cycle counts, not classification accuracy.

## Build and Run

```bash
cd examples/kws_cortex_m_int8
make release
make run
```

Sample output (CSV + summary):

```
name,weight_bytes,activation_bytes,cycles
qconv2d_3x3_8,...
qmaxpool2d_2x2,...
qdwconv2d_3x3,...
qpwconv2d_8x16,...
qglobal_avgpool2d,...
qdense_16x10,...
```

## Comparing Against the Float Runner

`weight_bytes` here counts the **external** int8 weight buffers, the int32 bias arrays, and the per-layer `Requantizer` tables. The float runner in [`examples/kws_cortex_m/`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m) reports `sizeof(layer)` for layers that embed their weights internally. Compare the totals at the bottom of each runner's summary block:

- int8 weights occupy 1 byte each versus 4 bytes for the float layers, so the quantized pipeline's flash footprint is **roughly 4x smaller** for the convolutional weights, plus a small fixed overhead for the `Requantizer` tables.
- Activation buffers are also 4x smaller (int8 vs float32). The peak activation tile shrinks accordingly.
- Cycle counts on a Cortex-M without an FPU are dramatically lower because the int8 path uses only integer ALU ops; on a Cortex-M with an FPU the gap narrows but the int8 MACs still benefit from CMSIS-NN-style SIMD on Cortex-M4 / M7 if you wire it up.

For the most direct comparison, run both runners on the same host with the same compiler and put the two CSV outputs side by side.

## Porting to a Cortex-M Target

1. **Cycle counter.** Compile with `-DTINYMIND_BENCH_CORTEX_M` so `bench::readCycleCounter()` reads `DWT->CYCCNT`.
2. **Output sink.** Replace `std::cout` with a UART wrapper (see the float runner's `port_stub.hpp` for the pattern — same `UartSink` shape works here).
3. **Drop calibration.** The deployable binary embeds pre-calibrated weight buffers and `Requantizer` tables as `const` data. Compile with `-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=0 -DTINYMIND_ENABLE_STD=0`. The forward path needs only `qaffine.hpp` and the `Q*` layer headers — no `<cmath>`, no float ops.
4. **Trained weights.** Load via assignment to `gW_*`, `gB_*`, and `gReq*` arrays produced by an offline calibration tool. The [`examples/pytorch/`](https://github.com/danmcleran/tinymind/tree/master/examples/pytorch) exporter pattern carries over directly; the only addition is per-tensor (or per-channel, for depthwise) scale fitting before serialization. The XOR example shows that for a small `QDense` chain.

## What This Example Demonstrates

- A complete MobileNet-style 2D pipeline runs end-to-end in **pure integer**.
- Per-channel depthwise calibration plus per-tensor calibration for everything else matches the TFLite reference shape exactly.
- The `quant_freestanding` corner of `unit_test/embedded` proves the same code builds with `FLOAT=0, STD=0, QUANT=1` — i.e. the deployable inference shape on a small MCU.
- The bench harness produces CSV that's directly diffable against the float runner's, so you can quote real numbers when you write up "we shrank flash by Nx and cycles by Mx by going to int8".

## See Also

- [Int8 Affine Quantization]({{ site.baseurl }}/architectures/int8-quantization) — the layer family, calibration helpers, and feature gate covered in depth.
- [Keyword Spotting CNN]({{ site.baseurl }}/getting-started/keyword-spotting-cnn) — the float reference runner this one mirrors.
- [PyTorch -> TinyMind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor) — smaller end-to-end example that walks training, calibration, weight emission, and inference.
