# `mixed_precision_kws`

Phase 16 mixed-precision exemplar — int8 CNN feature extractor → fp16
attention head → int8 dense classifier. Exercises the Phase 9 qbridge
converters (`affineI8ToFp16` / `fp16ToAffineI8`) plus the Phase 9
software `fp16_t` storage tier in production shape.

## Pipeline

```
input  [S=8][E=8]   float
   ----[ int8 frontend ]----------------------------
   QDense  E -> E (one call per sequence step)
   qrelu                                  -> [S][E] int8
   ----[ Phase 9 bridge: affineI8 -> fp16 ]---------
                                          -> [S][E] fp16
   ----[ fp16 attention head ]----------------------
   Linear (ReLU-kernel) self-attention with residual
   skip from the post-relu feature buffer, then
   mean-pool over S                       -> [E] fp16
   ----[ Phase 9 bridge: fp16 -> affineI8 ]---------
                                          -> [E] int8
   ----[ int8 classifier ]--------------------------
   QDense  E -> NUM_CLASSES               -> [NUM_CLASSES] int8 logits
```

The attention head's inner arithmetic runs in float promoted from
`fp16_t`. On targets that ship a vector fp16 ISA (NEON FEAT_FP16 via
`TINYMIND_ENABLE_SIMD_NEON_FP16=1`, AVX-512 fp16 extensions), the
promote-then-MAC pattern is a near-noop; on every other target the
scalar promote-store pair is the cost of admission for fp16 storage on
an MCU.

The Phase 9 bridges run scalar at the layer boundary by construction —
they are pointwise converters, never inner-loop primitives. The same
`affineI8ToFp16` / `fp16ToAffineI8` helpers also work for a Q-format /
int8 / Q-format chain.

## Precision tier per layer

| Layer                       | Storage    | Accumulator | Notes                       |
|-----------------------------|------------|-------------|-----------------------------|
| `QDense` (frontend)         | int8       | int32       | shared, applied per step    |
| `qrelu`                     | int8       | —           | clamp at zero_point         |
| `affineI8ToFp16Buffer`      | int8 → fp16| —           | Phase 9 bridge              |
| fp16 linear attention       | fp16       | float       | promote-MAC-store           |
| fp16 mean-pool              | fp16       | float       |                             |
| `fp16ToAffineI8Buffer`      | fp16 → int8| —           | Phase 9 bridge              |
| `QDense` (classifier)       | int8       | int32       |                             |

## Build + run

```
make             # debug build
make release     # -O3
make run         # parity report vs end-to-end float reference
make bench       # CSV cycle/byte report -> output/mixed_precision_kws.csv
make golden      # int8 logits for the 4-sample test set
```

`TINYMIND_ENABLE_FP16=1` is required (already set in the bundled
Makefile). With that gate off, `cpp/include/tinymind_fp16.hpp` is a
no-op and the build fails at the bridge call sites — the explicit gate
matches the Phase 9 design rule.

## Notes

The residual skip from the post-ReLU feature buffer into the attention
output keeps the pooled summary off the zero mean a tiny
linear-attention head would otherwise force on the bundled synthetic
inputs. Real KWS deployments have softmax classifiers and richer
encoders — the precision-tier pattern (int8 front + classifier
bracketing an fp16 head) is the load-bearing piece and survives every
substitution.
