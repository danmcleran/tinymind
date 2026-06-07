---
title: int8 ResNet Residual Block
parent: Examples
nav_order: 32
layout: default
---

# int8 ResNet Residual Block

A single int8 ResNet-style residual block stitched together from the Phase 10 composition ops and per-channel convolution. Demonstrates SAME-padding, a skip connection, and TFLite ADD semantics all on the int8 affine grid.

## How it works

- Pipeline: `QPad2D(1,1)` → `QConv2DPerChannel 3x3` → `qrelu` → `QPad2D(1,1)` → `QConv2DPerChannel 3x3`, then `QAdd` of that branch with the identity skip, then a closing `qrelu`, over a `4x4x2` input.
- `QPad2D` pads with each tensor's input zero_point so padded cells decode back to real-zero in the affine domain; it carries SAME padding around the VALID convolutions to preserve spatial dimensions. `QAdd` rescales the conv branch and the identity skip onto a shared output grid via `buildQAddParams` (TFLite ADD's left_shift + three multiplier/shift triples).
- Calibration is host-side: the block runs in float over an 8-sample synthetic dataset to collect activation ranges, then `computeAffineParamsAsymmetric` fits per-tensor activation params and `computePerChannelSymmetricScales` fits per-channel weight scales. Block end-to-end error is ~15% of output range — a four-stage int8 chain without QAT or cross-layer equalization picks up double-digit relative error; the pass threshold is 25%.

## Build and run

```bash
cd examples/resnet_block_int8
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for the host-side calibration). `make run` prints the per-tensor affine params and the worst max-abs error versus the float reference; `make plot` renders the parity overlay from `output/resnet_block_int8.csv`.

## Output

![int8 vs float parity for the ResNet residual block]({{ site.baseurl }}/assets/plots/resnet_block_int8.png)

Left panel overlays the float reference against the int8-dequantized output across all 32 output elements; the two tracks follow each other but visibly diverge at the peaks. Right panel plots the per-element absolute error, which tops out around 0.38 — consistent with the documented ~15% of output range for this un-equalized four-stage int8 chain.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/resnet_block_int8)
