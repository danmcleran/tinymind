---
title: MobileNetV2-shaped int8
layout: default
parent: Getting Started
nav_order: 8
---

# MobileNetV2-shaped int8

This tutorial walks the [`examples/mobilenetv2_int8/`](https://github.com/danmcleran/tinymind/tree/master/examples/mobilenetv2_int8) exemplar — a deterministic int8 MobileNetV2-shaped pipeline that exercises the inverted-residual block, linear bottlenecks, residual skips through `QAdd`, and the GAP + dense head. The build pattern in this file scales linearly to a full MobileNetV2-1.0 model (same block, 17× with the channel and stride schedule from the spec).

The exemplar ships a `make golden` mode — the int8 logit byte stream is locked by the `unit_test/integration/` Boost.Test suite, regardless of which SIMD backend the build resolves to.

## Pipeline (NHWC)

```
input  [16][16][4]
   QPad2D pad=1                            -> [18][18][4]
   QConv2DPerChannel 3x3 stride 2, F=8     -> [8][8][8]     (stem)
   qrelu
   ---- IR block 1 (stride 1, 8 -> 8, expand x4) -------------
   QPointwiseConv2D 8 -> 32                -> [8][8][32]    (expand)
   qrelu
   QPad2D pad=1                            -> [10][10][32]
   QDepthwiseConv2D 3x3 stride 1, C=32     -> [8][8][32]
   qrelu
   QPointwiseConv2D 32 -> 8                -> [8][8][8]     (project, linear)
   QAdd (skip from stem-relu)              -> [8][8][8]
   -----------------------------------------------------------
   ---- IR block 2 (stride 2, 8 -> 16, expand x4) -----------
   QPointwiseConv2D 8 -> 32                -> [8][8][32]
   qrelu
   QPad2D pad=1
   QDepthwiseConv2D 3x3 stride 2, C=32     -> [4][4][32]
   qrelu
   QPointwiseConv2D 32 -> 16               -> [4][4][16]    (project, linear, no skip)
   -----------------------------------------------------------
   QGlobalAvgPool2D                        -> [16]
   QDense 16 -> 4                          -> [4] int8 logits
```

## Three design rules at play

### 1. Linear bottlenecks

The 1x1 projection convolutions are **not** followed by `qrelu`. This is MobileNetV2's load-bearing design choice: the expand → depthwise → project trio keeps high-rank features inside the expanded space and projects back to the low-rank skip-friendly space **without a nonlinearity**. ReLU on a low-dimensional projection destroys information; the linear projection preserves it. MobileNetV3 and EfficientNet inherit this rule unchanged.

### 2. `qrelu` and `QGlobalAvgPool2D` are pass-throughs on the int8 grid

`qrelu` is a clamp at the input `zero_point` — no requantizer, no scale change. `QGlobalAvgPool2D` integer-averages over spatial axes, also no scale change. Consecutive layers reuse the upstream `(scale, zero_point)` rather than burning new requantizers. This is what makes the int8 grid composable without explosion: every requantizer in the pipeline corresponds to a real precision boundary (input change, weight scale collapse), not an artifact of layer chaining.

### 3. Per-channel weight scales where the math demands them

| Layer | Per-channel weights? |
|---|---|
| `QConv2DPerChannel` (stem) | Yes |
| `QPointwiseConv2D` (expand) | No (per-tensor) |
| `QDepthwiseConv2D` | Yes (TFLite mandate — depthwise channels have wildly varying magnitudes) |
| `QPointwiseConv2D` (project) | No (per-tensor) |
| `QDense` | No (per-tensor) |

Per-channel everywhere doubles requantizer storage with little accuracy gain in practice; the convention here matches what TFLite Micro and CMSIS-NN ship.

## Precision tier per layer

| Layer | Storage | Accumulator | Per-channel weights |
|---|---|---|---|
| `QPad2D` | int8 | — | n/a |
| `QConv2DPerChannel` (stem) | int8 | int32 | yes |
| `QPointwiseConv2D` (expand) | int8 | int32 | no |
| `QDepthwiseConv2D` | int8 | int32 | yes |
| `QPointwiseConv2D` (project) | int8 | int32 | no |
| `QAdd` | int8 | int32 | n/a |
| `QGlobalAvgPool2D` | int8 | int32 | n/a |
| `QDense` | int8 | int32 | no |

Every layer is pure integer at runtime. Calibration (`FLOAT=1 STD=1`) is host-only.

## Build and run

```sh
cd examples/mobilenetv2_int8
make             # debug
make release     # -O3
make run         # parity report (max-abs error vs float reference)
make bench       # CSV cycle/byte report -> output/mobilenetv2_int8.csv
make golden      # int8 logits for the bundled 4-sample test set
```

`make run` prints per-tensor affine params and the worst max-abs error vs the float reference; the bundled dataset passes within 50% of the logits range.

`make golden` writes a stable text dump of the int8 logit bytes that the integration suite asserts byte-for-byte. Because the SIMD backends' bit-exactness guarantee holds for every enabled backend, the same expected string passes regardless of which gate combination the example binary was built with.

## What the integration suite catches

[`unit_test/integration/`](https://github.com/danmcleran/tinymind/tree/master/unit_test/integration) shells out to this exemplar's `make golden` mode via `popen()` and compares the emitted byte stream to a baked-in expected string. Because the exemplar is deterministic (hand-crafted weights, fixed synthetic dataset, pure-integer forward), any silent drift in:

- the example pipeline,
- the `qaffine.hpp` requantizer,
- the `qcalibration.hpp` calibration helpers,
- or any SIMD specialization that claims bit-exactness,

trips the test. The root `Makefile`'s `check` target orders the integration suite after the example builds so the binaries always exist when the test runs.

## Where to take it next

The inverted-residual unit in this file is **the** load-bearing primitive of MobileNetV2, V3, and EfficientNet — same expand → DW → project shape, same linear-bottleneck rule. A full MobileNetV2-1.0 deployment is this block repeated 17 times with the channel and stride schedule baked into the spec, plus a stem and a classifier head. The build pattern scales linearly; no new primitive needed.

For a larger ResNet-shaped exemplar see [`examples/resnet18_block_int8/`](https://github.com/danmcleran/tinymind/tree/master/examples/resnet18_block_int8). For an attention-shaped one see [`examples/transformer_encoder_int8/`](https://github.com/danmcleran/tinymind/tree/master/examples/transformer_encoder_int8). For a mixed-precision (int8 + fp16) pipeline see [`examples/mixed_precision_kws/`](https://github.com/danmcleran/tinymind/tree/master/examples/mixed_precision_kws).

## See Also

- [Int8 Affine Quantization]({{ site.baseurl }}/architectures/int8-quantization) — the full layer family.
- [SIMD Backends]({{ site.baseurl }}/architectures/simd-backends) — the bit-exactness invariant the golden suite relies on.
- [PyTorch -> TinyMind int8 (importer)]({{ site.baseurl }}/getting-started/pytorch-importer) — how a real model's weights enter this shape.
- [Keyword Spotting CNN (int8)]({{ site.baseurl }}/getting-started/keyword-spotting-int8) — simpler depthwise-separable pipeline without inverted residuals.
