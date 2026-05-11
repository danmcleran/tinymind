# `resnet18_block_int8`

Phase 16 exemplar — int8 ResNet-18-shaped stem plus one basic-block stage,
exercised on a deterministic 4-sample synthetic dataset.

## Pipeline (NHWC)

```
input  [16][16][3]
   QPad2D pad=3                            -> [22][22][3]
   QConv2DPerChannel 7x7 stride 2, F=8     -> [8][8][8]
   qrelu  (clamp at p_stem.zero_point)
   QMaxPool2D 2x2 stride 2                 -> [4][4][8]
   ---- basic block (SAME padding, F=8) ---------------------
   QPad2D pad=1
   QConv2DPerChannel 3x3 stride 1, F=8     -> [4][4][8]
   qrelu
   QPad2D pad=1
   QConv2DPerChannel 3x3 stride 1, F=8     -> [4][4][8]
   QAdd (skip from post-pool path)         -> [4][4][8]
   qrelu                                   -> [4][4][8]
   ----------------------------------------------------------
   QGlobalAvgPool2D                        -> [8]
   QDense 8 -> 4                           -> [4] int8 logits
```

`QMaxPool2D` and `qreluBuffer` are pass-throughs on the int8 affine grid
(max and clamp do not rescale), so the post-pool grid is the post-stem
grid and the post-relu grid is the conv grid. `QGlobalAvgPool2D`
likewise shares its input scale and zero_point with its output (it is an
integer mean clamped to `[qmin, qmax]`).

## Precision tier per layer

| Layer                | Storage | Accumulator | Per-channel weights |
|----------------------|---------|-------------|---------------------|
| `QPad2D`             | int8    | —           | n/a                 |
| `QConv2DPerChannel`  | int8    | int32       | yes                 |
| `qrelu`              | int8    | —           | n/a                 |
| `QMaxPool2D`         | int8    | —           | n/a                 |
| `QAdd`               | int8    | int32       | n/a                 |
| `QGlobalAvgPool2D`   | int8    | int32       | n/a                 |
| `QDense`             | int8    | int32       | no (per-tensor)     |

Every layer is pure integer at runtime. Calibration (`FLOAT=1 STD=1`) is
host-only.

## Build + run

```
make             # debug build
make release     # -O3
make run         # parity report (max-abs err vs float reference)
make bench       # CSV cycle/byte report -> output/resnet18_block_int8.csv
make golden      # int8 logits for the 4-sample test set -> output/resnet18_block_int8.golden
```

The default `make run` prints per-tensor affine params and the worst
max-abs error vs the float reference; passes within 40 % of logits
range. The unit_test/integration suite consumes `make golden` output and
locks the int8 logit bytes byte-for-byte across SIMD gate combos.

## Notes

Weights are deterministic sinusoids — no PyTorch dependency. For a
trained model the same scaffolding plugs into the
`apps/import_pytorch/` flow from Phase 15: emit `weights.hpp` with the
same shapes, replace the weight-fill block with the imported tables,
keep every other line.
