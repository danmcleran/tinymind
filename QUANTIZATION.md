# Quantization Plan for TinyMind

Post-training int8 quantization (TFLite-style: per-tensor/per-channel scale + zero-point + int32 accumulators + integer requantization). Additive only — does not modify existing Q-format or float layers.

## Background: Q-format vs. quantization

These are distinct:

- **Q-format substitution** (current TinyMind): swap representation. Float `0.7531` → Q8.8 `0.7539` (nearest grid point). Resolution set by fractional bits. No calibration needed — just type swap. `QValue` carries no runtime metadata; `ValueType` is a single template parameter threaded through every layer.
- **Quantization** (this plan): map continuous distribution to discrete int8 levels via calibrated `scale` + `zero_point`. `q = round(x / scale) + zero_point`. Scale fit per tensor (or per channel) from observed min/max. Requires runtime metadata, requantization between layers, and int32 accumulators paired with int8 weights/activations.

## Design constraints

- Existing layers (`conv1d.hpp`, `conv2d.hpp`, `pool*.hpp`, `batchnorm.hpp`, `dropout.hpp`, `selfattention1d.hpp`, `fft1d.hpp`, `binarylayer.hpp`, `ternarylayer.hpp`, `neuralnet.hpp`) are all single-`ValueType` templates. **Do not modify them.**
- `QValue` (`qformat.hpp`) has no runtime scale/zero-point and no accumulator-type separation. **Do not modify it.**
- Add a parallel quantization path. User opts in via type choice + new feature gate.
- Preserve the four-corner `(FLOAT, STD)` embedded regression matrix; add a fifth axis `QUANT`.

## New feature gate

In `cpp/include/tinymind_platform.hpp`:

- `TINYMIND_ENABLE_QUANTIZATION` (default `0`) — gates new quantization headers entirely. Embedded freestanding builds untouched unless flipped.

Inference-only quantized build target: `TINYMIND_ENABLE_QUANTIZATION=1`, `TINYMIND_ENABLE_FLOAT=0`, `TINYMIND_ENABLE_STD=0`. Calibration tooling lives behind `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD` so host-only.

## Phase 1 — Core types (`cpp/qaffine.hpp`, new file)

```cpp
template<typename StorageType, typename AccumulatorType>
struct QAffineTensor {
    // runtime metadata, per-tensor (extend later to per-channel)
    float scale;
    StorageType zero_point;
    // raw storage held by caller (layer owns weight buffers)
};

template<typename SrcAccum, typename DstStorage>
struct Requantizer {
    int32_t multiplier;   // fixed-point M0 (Q0.31)
    int8_t  shift;        // right shift after multiply
    DstStorage zero_point;
    static DstStorage apply(SrcAccum acc);  // saturating round
};
```

- int8 activations + int8 weights + int32 accumulator (TFLite reference kernel shape).
- Requantization done as int multiply + shift (no float in inference path) — matches CMSIS-NN style.
- Calibration helpers (float → scale/zp + multiplier/shift decomposition) live behind `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD`. Inference-only build gets no float pull-in.

## Phase 2 — Quantized layers (parallel to existing)

New headers, mirror existing API but template on `<InputType, WeightType, AccumType, OutputType>`:

- `cpp/qconv2d.hpp` — `QConv2D` with int32 accumulator, requantize on output
- `cpp/qdepthwiseconv2d.hpp` — `QDepthwiseConv2D` (per-channel scale critical here — TFLite mandates)
- `cpp/qpointwiseconv2d.hpp` — `QPointwiseConv2D`
- `cpp/qpool2d.hpp` — `QMaxPool2D` / `QAvgPool2D` (avg needs requantize, max is rescale-free if same scale)
- `cpp/qdense.hpp` — quantized fully-connected (standalone, not coupled to `NeuralNet<>`)

Each layer carries its own `Requantizer` in member state. No changes to non-`Q*` siblings.

## Phase 3 — Weight import path

Leave `ValueConverter` in `cpp/include/nnproperties.hpp` alone. Add separate:

- `cpp/include/qcalibration.hpp` — float → int8 quantization with per-tensor or per-channel scale fitting (min/max + symmetric/asymmetric). Host-only (`TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD`).
- Helper to compute requant `(multiplier, shift)` from `(input_scale * weight_scale) / output_scale`.

PyTorch import path in `examples/pytorch/` gets a sibling: `examples/pytorch_quant/` showing quantize-aware weight export. Existing example untouched.

## Phase 4 — Activation functions

- Quantized ReLU = clamp at zero-point.
- Quantized sigmoid/tanh = lookup table (similar pattern to `lookupTables.cpp` but indexed by int8 input).
- Add `cpp/qactivations.hpp`. Existing `activationFunctions.hpp` untouched.

## Phase 5 — Tests + bench

- `unit_test/quantization/` — new Boost.Test suite. Verify:
  - `QConv2D` matches float reference within tolerance after quantize/dequantize round-trip
  - `Requantizer` round-trip
  - Per-channel depthwise correctness
- `unit_test/embedded/` regression matrix grows: add `QUANT=1` corner. Confirms quantization path builds with `STD=0, FLOAT=0` (inference-only embedded).
- `examples/kws_cortex_m/` gets quant variant `kws_cortex_m_int8/` — same pipeline, int8 weights, CSV cycle/byte report for comparison.

## Phase 6 — Docs

- Append section to `CLAUDE.md` under Architecture Overview: "Quantization (optional, `TINYMIND_ENABLE_QUANTIZATION=1`)".
- README gets one paragraph + pointer to example.

## Risk / non-goals

- **No QAT** (quantization-aware training). Post-training quantization only — fits existing "PyTorch float training → MCU inference" workflow.
- **No int4 / mixed precision** initially. Just int8 weights/activations + int32 accumulator.
- **No fusion pass** (conv+bn+relu folded). Each layer standalone first; fusion later.
- **Per-channel only on depthwise** in v1 (TFLite minimum requirement). Other layers per-tensor.

## Order of work (suggested PRs)

1. `tinymind_platform.hpp` gate + `qaffine.hpp` types + unit test for `Requantizer`
2. `qcalibration.hpp` + tests
3. `qdense.hpp` + `qactivations.hpp` (smallest end-to-end slice)
4. `qconv2d.hpp` + `qpool2d.hpp`
5. `qdepthwiseconv2d.hpp` + `qpointwiseconv2d.hpp` (per-channel)
6. KWS int8 example + bench comparison
7. Embedded regression matrix update + docs

Each PR self-contained, existing tests stay green throughout (additive only).

## Codebase survey results (reference)

From investigation 2026-05-08:

- All composable layers parameterize on single `typename ValueType`. None separate input/weight/accum/output types.
- `cpp/neuralnet.hpp:2476` — `ValueType sum` accumulator (single type) in forward pass.
- `cpp/conv2d.hpp:113,126` — `ValueType sum = bias; sum += weight * input;` (same type for MAC and accumulation).
- `cpp/qformat.hpp:383–408` — `QValue` is pure compile-time fixed/fractional bit split. Single union storage. No runtime metadata field.
- No `scale`, `zero_point`, `requant` symbols exist outside compile-time / per-stage shifts (FFT butterfly, dropout `1/(1-p)`, gelu scale).
- `cpp/include/nnproperties.hpp:155–265` — `ValueConverter` is the only existing host→target weight conversion path. Used by `NetworkPropertiesFileManager`.
- `cpp/include/tinymind_traits.hpp` — `enable_if`, `is_floating_point` available for SFINAE at `STD=0`.

These confirm: quantization cannot retrofit cleanly into the existing single-`ValueType` templates without invasive changes. Parallel layer set is the right call.
