# KWS-style Cortex-M example (int8 quantized)

Same pipeline as `examples/kws_cortex_m`, with every layer replaced by
the quantized variant from `cpp/q*.hpp`. Demonstrates an end-to-end
post-training int8 path: weights and activations are int8, accumulators
are int32, and a per-layer integer Requantizer (Q0.31 multiplier + shift)
rescales between layers without any float arithmetic on the inference
path.

## Pipeline

```
input [20 x 20 x 1] int8
  -> QConv2D 3x3, 8 filters           -> [18 x 18 x 8]  int8
  -> QMaxPool2D 2x2                   -> [9  x 9  x 8]  int8
  -> QDepthwiseConv2D 3x3 (per-chan)  -> [7  x 7  x 8]  int8
  -> QPointwiseConv2D 8 -> 16         -> [7  x 7  x 16] int8
  -> QGlobalAvgPool2D                 -> [16]           int8
  -> QPointwiseConv2D (dense) 16 -> 10 (int8 logits)
```

`QDepthwiseConv2D` carries a per-channel `Requantizer` array, matching
TFLite's mandate. Other layers use a single per-tensor Requantizer.

## Build and run

```bash
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

## Comparing against the float runner

`weight_bytes` here counts the external int8 weight buffers, the int32
bias arrays, and the per-layer Requantizer tables. The float runner in
`examples/kws_cortex_m/` reports `sizeof(layer)` for layers that embed
their weights internally. Compare the totals at the bottom of each
runner's summary block — int8 weights occupy 1 byte each versus 4 bytes
for the float layers, so the quantized pipeline's flash footprint is
roughly 4x smaller for the convolutional weights, plus a small fixed
overhead for the Requantizer tables.

## Calibration

The host runner generates synthetic float weights and converts them
with the helpers in `cpp/include/qcalibration.hpp`:

* `quantize` / `quantizeBuffer` — symmetric int8 weight quantization
* `computePerChannelSymmetricScales` — per-channel scale fit for
  the depthwise weights
* `buildRequantizer` — combines `(input_scale, weight_scale, output_scale,
  output_zero_point)` into the integer `(multiplier, shift)` pair the
  Requantizer consumes at runtime

A real deployment would feed observed activation ranges from a
representative dataset through `RangeObserver` /
`computeAffineParamsAsymmetric` to fit per-tensor activation scales.
This runner uses fixed scales (`1/127` for activations, `0.1/127` for
weights) since the demo cares about footprint and cycle counts, not
accuracy.

## Porting to a Cortex-M target

1. **Cycle counter.** Compile with `-DTINYMIND_BENCH_CORTEX_M` so
   `bench::readCycleCounter()` reads `DWT->CYCCNT`.
2. **Output sink.** Replace `std::cout` with a UART wrapper (see the
   float runner's `port_stub.hpp`).
3. **Drop calibration.** The deployable binary embeds pre-calibrated
   weight buffers and Requantizer tables as `const` data. Compile with
   `-DTINYMIND_ENABLE_QUANTIZATION=1 -DTINYMIND_ENABLE_FLOAT=0
   -DTINYMIND_ENABLE_STD=0`. The forward path needs only `qaffine.hpp`
   and the Q* layer headers — no `<cmath>`, no float ops.
4. **Trained weights.** Load via assignment to `gW_*`, `gB_*`, and
   `gReq*` arrays produced by an offline calibration tool. The
   `examples/pytorch/` exporter pattern carries over directly; the
   only addition is per-tensor (or per-channel) scale fitting before
   serialization.
