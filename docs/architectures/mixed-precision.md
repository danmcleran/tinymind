---
title: Mixed Precision
layout: default
parent: Architectures
nav_order: 8
---

# Mixed Precision

TinyMind composes its three numeric pipelines through a small set of **pointwise converters** that live at layer boundaries, plus a software half-precision storage tier. A single network can run an int8 affine CNN frontend, hand off to a Q-format LSTM head, hand off again to an fp16 attention block, and project back to int8 for the classifier — every layer keeps the runtime cost of its own grid, the bridges only run once per tensor crossing.

## Three pipelines, one model

| Pipeline | Storage | Where it lives | When it wins |
|---|---|---|---|
| `QValue` Q-format | int8 / int16 / int32 / int64 with a compile-time binary point | `cpp/qformat.hpp` + `cpp/neuralnet.hpp` | Trainable on-MCU, single global grid, no per-tensor metadata |
| Float | `float` / `double` | Same templates, different `ValueType` | Host development, training |
| Int8 affine | int8 weights + int8 activations + per-tensor `(scale, zero_point)` | `cpp/q*.hpp` family | TFLite-shape inference, multi-grid (each tensor picks its own range) |

The qbridge converters tie the three together. The `simd_neon_fp16.hpp` backend adds vector specializations for fp16 storage on Arm hardware that supports it; this page covers the storage tier and the converters.

## qbridge.hpp — pointwise converters

[`cpp/qbridge.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/qbridge.hpp) provides single-value and buffer-batch converters at layer boundaries. Float at runtime, no `<cmath>` (rounding via sign-aware cast). Gated on `TINYMIND_ENABLE_FLOAT`; freestanding-safe at `STD=0`.

### Int8 affine ↔ float

| Helper | Direction |
|---|---|
| `affineDequantize<Src>(q, scale, zp)` | int8 affine → float |
| `affineQuantize<Dst>(x, scale, zp, qmin, qmax)` | float → int8 affine |
| `affineDequantizeBuffer(src, dst, n, scale, zp)` | buffer |
| `affineQuantizeBuffer(src, dst, n, scale, zp, qmin, qmax)` | buffer |

### Q-format ↔ float

| Helper | Direction |
|---|---|
| `qValueToFloat<QV>(q)` | `QValue<Q, F>` → float |
| `floatToQValue<QV>(x)` | float → `QValue<Q, F>` |

### Q-format ↔ int8 affine

| Helper | Direction |
|---|---|
| `qValueToAffine<QV, Dst>(q, scale, zp, qmin, qmax)` | `QValue` → int8 affine |
| `affineToQValue<QV, Src>(q, scale, zp)` | int8 affine → `QValue` |

### Half-precision (gated on `TINYMIND_ENABLE_FP16=1`)

| Helper | Direction |
|---|---|
| `affineI8ToFp16(q, scale, zp)` | int8 affine → fp16 |
| `fp16ToAffineI8(h, scale, zp, qmin, qmax)` | fp16 → int8 affine |
| `affineI8ToBf16(q, scale, zp)` | int8 affine → bf16 |
| `bf16ToAffineI8(h, scale, zp, qmin, qmax)` | bf16 → int8 affine |

Buffer-batch versions of every variant follow the same naming.

## fp16_t and bf16_t storage tier

[`cpp/include/tinymind_fp16.hpp`](https://github.com/danmcleran/tinymind/blob/master/cpp/include/tinymind_fp16.hpp) provides software-only `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) storage structs wrapping `uint16_t`. The conversion helpers (`floatToFp16` / `fp16ToFloat`, `floatToBf16` / `bf16ToFloat`) handle normals, subnormals, Inf, and NaN.

This is a **storage tier**, not an arithmetic tier. The structs are 16-bit; arithmetic happens by promoting to float at the call site. On targets that ship vector fp16 arithmetic (NEON FEAT_FP16 via `TINYMIND_ENABLE_SIMD_NEON_FP16=1`, AVX-512 fp16 extensions) the promote-then-MAC pattern is a near-noop. On every other target the scalar promote-store pair is the cost of admission for fp16 storage on an MCU.

Gates:

- `TINYMIND_ENABLE_FP16=1` — pulls in the storage types and the conversion helpers.
- Conversion helpers additionally require `TINYMIND_ENABLE_FLOAT=1`.

The `unit_test/embedded/Makefile` exercises this corner as `fp16_freestanding` (`FLOAT=1 FP16=1 QUANT=1 STD=0`) to confirm the half-precision and bridge headers stay freestanding-clean.

## Mixed-precision exemplar — `mixed_precision_kws`

[`examples/mixed_precision_kws/`](https://github.com/danmcleran/tinymind/tree/master/examples/mixed_precision_kws) wires the qbridge converters in production shape:

```
input  [S=8][E=8]   float
   ----[ int8 frontend ]----------------------------
   QDense  E -> E (one call per sequence step)
   qrelu                                  -> [S][E] int8
   ----[ qbridge: affineI8 -> fp16 ]----------------
                                          -> [S][E] fp16
   ----[ fp16 attention head ]----------------------
   Linear (ReLU-kernel) self-attention with residual
   skip from the post-relu feature buffer, then
   mean-pool over S                       -> [E] fp16
   ----[ qbridge: fp16 -> affineI8 ]----------------
                                          -> [E] int8
   ----[ int8 classifier ]--------------------------
   QDense  E -> NUM_CLASSES               -> [NUM_CLASSES] int8 logits
```

The precision-tier pattern — int8 front + classifier bracketing an fp16 head — is the load-bearing piece. Real KWS deployments have softmax classifiers and richer encoders; the int8 / fp16 / int8 sandwich survives every substitution.

## When to bridge

- **int8 → fp16 → int8 around an attention block.** Linear self-attention has an inner `Q' KV` matmul whose dynamic range is hard to pin down at calibration time without losing accuracy. fp16 in the middle absorbs the range, the surrounding int8 keeps storage and the conv MACs cheap.
- **int8 → float → int8 around a softmax.** When deploying on a target with float MACs but limited integer throughput, the softmax can run in float between two int8 layers without disturbing the deployable shape.
- **Q-format ↔ int8 within a hybrid model.** When migrating an existing `QValue`-based network to int8 incrementally, `qValueToAffine` / `affineToQValue` let you swap one layer at a time and validate parity at each step.

## What this is not

- **Not QAT.** Mixed precision is a deployment story, not a training story.
- **Not fp16 arithmetic.** The library treats fp16 as a storage tier; inner arithmetic promotes to float. The vector fp16 ISA gates (`SIMD_NEON_FP16`, AVX-512 fp16) get there on hardware that supports it, but the library does not synthesize fp16 software arithmetic.
- **Not int4.** Storage is int8 / int16 / int32 / fp16 / bf16 / float / double. Sub-byte storage is out of scope.

## See Also

- [Int8 Affine Quantization]({{ site.baseurl }}/architectures/int8-quantization) — the int8 grid the converters cross.
- [Q-Format (Fixed-Point)]({{ site.baseurl }}/q-format) — the `QValue` grid the converters also cross.
- [SIMD Backends]({{ site.baseurl }}/architectures/simd-backends) — `SIMD_NEON_FP16` is the vector fp16 path on hardware that supports it.
- [`examples/mixed_precision_kws/`](https://github.com/danmcleran/tinymind/tree/master/examples/mixed_precision_kws) — the end-to-end exemplar.
