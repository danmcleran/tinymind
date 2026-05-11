# resnet_block_int8

Phase 10 demonstration: int8 ResNet-style residual block.

## What it shows

A single residual block stitched together from the Phase-10 composition
ops and per-channel `QConv2DPerChannel`:

```
input  [4][4][2]
   │
   ├── QPad2D(1,1) ── QConv2DPerChannel(3x3, per-channel) ── qrelu
   │                                                          │
   │                                                          QPad2D(1,1)
   │                                                          │
   │                                                          QConv2DPerChannel(3x3, per-channel)
   │                                                          │
   └─────────────────────────────────────► QAdd ◄─────────────┘
                                            │
                                          qrelu
                                            │
                                       output [4][4][2]
```

`QPad2D` carries SAME padding around the VALID `QConv2DPerChannel` to
preserve spatial dimensions through both convolutions; pad value is set
to each tensor's input zero point so the padded cells decode back to
real-zero in the affine domain. `QAdd` rescales the conv2 branch and the
identity skip onto a shared output grid using TFLite ADD semantics
(`buildQAddParams`). The closing `qrelu` clamps at the sum-tensor zero
point.

## How it runs

The driver hand-builds small but non-trivial weights, runs the block in
float over an 8-sample synthetic dataset to collect activation ranges,
calibrates per-tensor activation params (`computeAffineParamsAsymmetric`)
and per-channel symmetric weight scales (`computePerChannelSymmetricScales`),
quantizes weights and biases, builds per-channel Requantizers, then runs
the block end-to-end on int8. Output is dequantized and compared
against the float reference.

## Numbers

After `make run`:

- conv1 single-layer max-abs error: ~0.5 % of conv1 range
- block end-to-end max-abs error: ~15 % of output range (8 samples)

Pass threshold is 25 % of the dynamic range — generous because a
4-stage int8 chain without QAT or cross-layer equalization will
naturally pick up double-digit relative error. Phase 15 (calibration
upgrades: percentile, KL, conv+bn fold, CLE) tightens this.

## Files

- `resnet_block_int8.cpp` — driver. Self-contained; no external weight
  file. Uses `QConv2DPerChannel`, `QPad2D`, `QAdd`, `qrelu`, plus
  calibration helpers from `cpp/include/qcalibration.hpp`.
- `Makefile` — builds with `TINYMIND_ENABLE_FLOAT=1 STD=1 QUANTIZATION=1`.

## Build and run

```sh
make clean && make && make run
```
