---
title: int8 MobileNetV2 Pipeline
parent: Examples
nav_order: 34
layout: default
---

# int8 MobileNetV2 Pipeline

A int8 MobileNetV2-shaped pipeline: two inverted-residual blocks (one stride-1 with a residual skip, one stride-2 without) wired around a stride-2 stem and a GAP + dense head. Demonstrates the inverted-residual unit and MobileNetV2's linear-bottleneck rule on the int8 affine grid.

## How it works

- Pipeline (NHWC, `16x16x4` input): `QPad2D` → `QConv2DPerChannel 3x3 s=2` stem → `qrelu`, then IR block 1 (`QPointwiseConv2D 8→32` expand → `qrelu` → `QDepthwiseConv2D 3x3 s=1` → `qrelu` → `QPointwiseConv2D 32→8` project → `QAdd` skip), then IR block 2 (expand → DW 3x3 s=2 → project 32→16, no skip), then `QGlobalAvgPool2D` → `QDense 16→4` (int8 logits).
- Linear bottlenecks: the 1x1 projection convolutions are deliberately **not** followed by `qrelu`, matching MobileNetV2's design rule (expand → DW → project keeps high-rank features in the expanded space and projects back without a nonlinearity). `QDepthwiseConv2D` is per-channel weight-scaled (TFLite mandate); `qrelu` and `QGlobalAvgPool2D` are pass-throughs that reuse the upstream `(scale, zero_point)`.
- Every layer is pure integer at runtime with int32 accumulators; calibration (`FLOAT=1 STD=1`) is host-only. `make run` passes within 50% of logits range.

## Build and run

```bash
cd examples/mobilenetv2_int8
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1`. Extra targets:

- `make bench` — CSV cycle/byte report to `output/mobilenetv2_int8.csv`
- `make golden` — int8 logits for the 4-sample test set to `output/mobilenetv2_int8.golden` (consumed by `unit_test/integration`, locked byte-for-byte across SIMD gate combos)

## Output

![int8 vs float parity for the MobileNetV2 pipeline]({{ site.baseurl }}/assets/plots/mobilenetv2_int8_parity.png)

Left panel overlays the float reference against the int8-dequantized logits across the 4 output classes — the two tracks track the same monotone trend with a small gap at the extremes. Right panel shows the per-element absolute error staying under ~0.007, comfortably inside the 50% pass threshold for this two-inverted-residual-block pipeline.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/mobilenetv2_int8)
