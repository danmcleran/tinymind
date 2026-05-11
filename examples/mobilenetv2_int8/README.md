# `mobilenetv2_int8`

Phase 16 exemplar — int8 MobileNetV2-shaped pipeline. Two inverted-residual
blocks (one stride-1 with a residual skip, one stride-2 without), wired
around a 3x3 stride-2 stem and a GAP + dense head.

## Pipeline (NHWC)

```
input  [16][16][4]
   QPad2D pad=1                            -> [18][18][4]
   QConv2DPerChannel 3x3 stride 2, F=8     -> [8][8][8]     (stem)
   qrelu
   ---- IR block 1 (stride 1, 8 -> 8, expand x4) -------------
   QPointwiseConv2D 8 -> 32                -> [8][8][32]    (expand)
   qrelu
   QPad2D pad=1
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

Linear bottlenecks: the 1x1 projection convolutions are **not** followed
by `qrelu`, matching MobileNetV2's "linear bottleneck" design rule (the
expand→DW→project trio keeps high-rank features inside the expanded
space and projects back to the low-rank skip-friendly space without a
nonlinearity).

`qrelu` and `QGlobalAvgPool2D` are pure pass-throughs on the int8 affine
grid (clamp and integer-mean), so consecutive layers reuse the upstream
`(scale, zero_point)`.

## Precision tier per layer

| Layer                       | Storage | Accumulator | Per-channel weights |
|-----------------------------|---------|-------------|---------------------|
| `QPad2D`                    | int8    | —           | n/a                 |
| `QConv2DPerChannel` (stem)  | int8    | int32       | yes                 |
| `QPointwiseConv2D` (expand) | int8    | int32       | no (per-tensor)     |
| `QDepthwiseConv2D`          | int8    | int32       | yes (TFLite mandate)|
| `QPointwiseConv2D` (project)| int8    | int32       | no (per-tensor)     |
| `QAdd`                      | int8    | int32       | n/a                 |
| `QGlobalAvgPool2D`          | int8    | int32       | n/a                 |
| `QDense`                    | int8    | int32       | no (per-tensor)     |

Every layer is pure integer at runtime. Calibration (`FLOAT=1 STD=1`) is
host-only.

## Build + run

```
make             # debug build
make release     # -O3
make run         # parity report (max-abs err vs float reference)
make bench       # CSV cycle/byte report -> output/mobilenetv2_int8.csv
make golden      # int8 logits for the 4-sample test set -> output/mobilenetv2_int8.golden
```

Default `make run` prints per-tensor affine params and the worst max-abs
error vs the float reference; passes within 50 % of logits range. The
`unit_test/integration` suite consumes `make golden` output and locks
the int8 logit bytes byte-for-byte across SIMD gate combos.

## Notes

The inverted-residual unit is the load-bearing primitive of MobileNetV2,
V3, and EfficientNet — same expand→DW→project shape, same linear
bottleneck rule. A full MobileNetV2-1.0 model is the same block
repeated 17 times with the channel and stride schedule baked into the
spec; the build pattern in this file scales linearly.
