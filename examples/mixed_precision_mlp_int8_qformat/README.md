# `mixed_precision_mlp_int8_qformat`

Phase 17 hybrid mixed-precision exemplar — int8 affine MLP frontend → Q-format
hidden tier → int8 affine classifier, with the **pure-integer** Phase 17
qbridge converters in between. The end-to-end inference path has no
floating-point math at runtime; the deployable target shape is
`TINYMIND_ENABLE_QUANTIZATION=1, TINYMIND_ENABLE_FLOAT=0, TINYMIND_ENABLE_STD=0`.

## Pipeline

```
input [E_IN=6]   float
   ----[ host-side calibration: quantize to int8 affine ]----
                                                  -> [E_IN] int8
   ----[ int8 frontend ]------------------------------------
   QDense E_IN -> E_HIDDEN
   qrelu                                          -> [E_HIDDEN] int8
   ----[ Phase 17 bridge: affineToQValueIntBuffer ]---------
   pure-integer multiply-by-quantized-multiplier
                                                  -> [E_HIDDEN] Q8.8
   ----[ Q-format hidden ]----------------------------------
   Q8.8 dense matvec (int32 accumulator -> Q8.8) -> [E_HIDDEN] Q8.8
   ----[ Phase 17 bridge: qValueToAffineIntBuffer ]---------
   pure-integer multiply-by-quantized-multiplier
                                                  -> [E_HIDDEN] int8
   ----[ int8 classifier ]----------------------------------
   QDense E_HIDDEN -> E_OUT                       -> [E_OUT] int8 logits
```

## Why hybrid int8 + Q-format

A common embedded-inference deployment story:

* the model is trained in PyTorch / TensorFlow with float weights;
* the outermost layers go through standard post-training int8 quantization
  (TFLite / ONNX QDQ); but
* an existing TinyMind Q-format implementation already runs on the target
  (e.g. a `NeuralNet<Q8.8>` hand-tuned for the MCU); or
* a particular hidden layer wants Q-format's compile-time fixed/fractional
  bit split (no per-tensor `scale` field, no runtime requantizers) for
  tighter cycle budgets.

The Phase 17 pure-integer qbridge (`cpp/qbridge.hpp`) is the missing piece:
it converts between the two precision tiers without `<cmath>`, `<type_traits>`,
or `float`, using only the same Q0.31 `(multiplier, shift)` primitive that
`Requantizer` already uses.

The conversion params are built host-side once, at calibration time, by
`buildAffineToQValueIntParams<QV>` and `buildQValueToAffineIntParams<QV>`
(both gated on `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD`). The
resulting integer triples ship to the target alongside the rest of the
quantized model; the target runs `affineToQValueInt` / `qValueToAffineInt`
pure-integer.

## Build & run modes

```bash
cd examples/mixed_precision_mlp_int8_qformat
make            # build
make run        # parity report vs float reference
make bench      # CSV cycle/byte report
make golden     # int8 byte stream for the bundled deterministic test set
```

`make run` reports the max-abs error of the int8 logits vs the float
reference; on the bundled synthetic dataset it lands around `0.005` (well
below the 60 %-of-output-range tolerance the exemplar sets).

## Importing from PyTorch / TensorFlow

`apps/import_pytorch/tinymind_import.py` carries a `QFormatDense`
descriptor and a `HybridBoundary` declaration that emit the same
`weights.hpp` shape this exemplar uses. The boundaries list captures
each precision-tier transition; the emitter writes the precomputed
`(multiplier, shift, zero_point)` triples directly into the header so
the deployable shape needs no host-side helper call at startup.

TensorFlow models reach the same emitter via `tf2onnx` plus the QDQ
importer in `apps/import_onnx/` — see `apps/import_onnx/README.md` for
the recipe.
