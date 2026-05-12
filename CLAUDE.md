# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build and run all tests and examples
make check

# Build and run individual test suites
cd unit_test/nn && make clean && make && make run
cd unit_test/qformat && make clean && make && make run
cd unit_test/qlearn && make clean && make && make run

# Run a single test binary directly
cd unit_test/nn/output && ./nn_unit_test

# Build examples individually
cd examples/xor && make clean && make
cd examples/maze && make clean && make
cd examples/dqn_maze && make clean && make
```

Compiler flags: `-Wall -Wextra -Werror -Wpedantic` (debug builds also add `-ggdb`; release adds `-O3`).

The `BOOST_HOME` environment variable must be set to the Boost installation path (used by unit test Makefiles).

### Platform Feature Gates

Preprocessor macros in `cpp/include/tinymind_platform.hpp` control optional hosted dependencies and per-ISA SIMD specializations. **All default to 0** (freestanding, scalar); hosted Makefiles set `=1` selectively.

- `TINYMIND_ENABLE_FLOAT` — `float`/`double` as `ValueType`
- `TINYMIND_ENABLE_STD` — `<cmath>`, `<type_traits>`, `namespace std::`
- `TINYMIND_ENABLE_HOSTED_IO` — `<fstream>`, `<vector>`, `<cstdlib>`, `<cstdio>` (file weight serialization)
- `TINYMIND_ENABLE_OSTREAMS` — `<ostream>` and `QValue::operator<<`
- `TINYMIND_ENABLE_HOSTED_RAND` — `<cstdlib>` `rand()` (Dropout training mode, ScheduledSampling, Xavier)
- `TINYMIND_ENABLE_QUANTIZATION` — `cpp/q*.hpp` int8 quantization layer family + `Requantizer`. Calibration helpers in `cpp/include/qcalibration.hpp` are additionally gated on `FLOAT && STD` (host-only). Existing fixed-point and float pipelines are unaffected.
- `TINYMIND_ENABLE_FP16` — Phase 9 half-precision storage tier: `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) in `cpp/include/tinymind_fp16.hpp`, with float conversion helpers. Storage tier; Phase 14's `simd_neon_fp16.hpp` provides the matching vector specialization. Conversion helpers additionally require `TINYMIND_ENABLE_FLOAT`.
- `TINYMIND_ENABLE_INT16_ACCUM` — Phase 12 wide carry-state for `QLSTMCell` (cell-state stored as `int16_t` rather than `int8_t`). The cell template itself only needs the C++ type, so the gate purely controls whether the embedded regression matrix advertises the int16 corner; targets without an int16 storage path simply do not instantiate the wide template.
- `TINYMIND_ENABLE_SIMD_NEON` / `_NEON_DOTPROD` / `_NEON_FP16` / `_SVE` / `_SVE2` / `_HELIUM_MVE_I` / `_HELIUM_MVE_F` / `_AVX2` / `_AVX_VNNI` / `_AVX512F` / `_AVX512_VNNI` — Phase 14 ISA-capability SIMD gates. Gates name ISA extensions, never CPU models. Headers in `cpp/include/simd/` open with a `static_assert` enforcing Arm's prerequisite chain (DOTPROD requires NEON; SVE/SVE2/FP16 vector forms require NEON; AVX_VNNI requires AVX2; AVX512_VNNI requires AVX512F; Helium MVE_I/MVE_F are M-profile only and mutually exclusive with NEON/SVE). Scalar fallback is byte-identical to the pre-Phase-14 build when every gate is off.
- `TINYMIND_ENABLE_OPENMP` — Phase 14 OpenMP parallelization of the output-filter loop in `QConv2D` / `QConv2DPerChannel` via `TINYMIND_PARALLEL_FOR_OUTER` in `cpp/include/threading.hpp`. Orthogonal to every SIMD gate; caller passes `-fopenmp` separately.

`unit_test/embedded` is the regression guard — its Makefile builds eight corners of the `(FLOAT, STD, QUANT, FP16, INT16_ACCUM, SIMD_*)` matrix (`freestanding`, `no_stdlib`, `no_fpu`, `hosted`, `quant_freestanding`, `fp16_freestanding`, `int16_accum_freestanding`, `simd_disabled`) and `make check` runs all of them. A separate `simd_prereq_regressions` make target locks the static_assert prerequisite chain via compile-failure checks. Float-typed Adam/RMSprop/Xavier require both `FLOAT && STD`. When adding a new header to `cpp/`, include `"include/tinymind_platform.hpp"` first and gate any `<cmath>`/`<type_traits>`/`std::` use; for SFINAE on `is_floating_point`, use `tinymind::enable_if` and `tinymind::is_floating_point` from `cpp/include/tinymind_traits.hpp` so the layer compiles at `STD=0`.

## Architecture Overview

TinyMind is a **header-only C++ template library** for neural networks and Q-learning, designed for embedded systems with no FPU, GPU, or vectorized instruction requirements. The design is inspired by Andrei Alexandrescu's policy-based design from *Modern C++ Design*.

### Core Library (`cpp/`)

- **`qformat.hpp`** — Fixed-point arithmetic templates. Supports Q8.8, Q16.16, Q24.8, Q32.32, and other formats (up to 128-bit). Recently extended to also support `float` as a value type.
- **`neuralnet.hpp`** — The main neural network template (~3200 lines). Supports feed-forward and recurrent networks. Network topology, value type, activation function, and other policies are all template parameters.
- **`qlearn.hpp`** — Q-learning and DQN reinforcement learning implementation.
- **`activationFunctions.hpp`**, **`fixedPointTransferFunctions.hpp`** — Activation functions (sigmoid, tanh, ReLU, etc.) with separate implementations for fixed-point vs. float.
- **`lookupTables.cpp`** — Large pre-computed lookup tables for fixed-point activation functions (sigmoid, tanh, exp, log, sin, cos). This is the only `.cpp` file in the core library; everything else is headers.
- **`cpp/include/nnproperties.hpp`** — Defines the property/policy classes used to configure neural network templates (layer sizes, learning rates, activation functions, etc.).

### Standalone Composable Layers (`cpp/`)

These layers sit outside the neural network template and can be chained into pipelines:

- **`conv1d.hpp`** — 1D convolution layer for time-series feature extraction.
- **`pool1d.hpp`** — `MaxPool1D` and `AvgPool1D` for downsampling.
- **`conv2d.hpp`** — 2D convolution for spectrograms/images, NHWC layout, VALID padding.
- **`depthwiseconv2d.hpp`** / **`pointwiseconv2d.hpp`** — MobileNet-style depthwise-separable blocks.
- **`pool2d.hpp`** — `MaxPool2D`, `AvgPool2D`, and `GlobalAvgPool2D` (GAP replaces the big flatten-to-dense matrix).
- **`selfattention1d.hpp`** — Linear self-attention layer using ReLU kernel feature map (no softmax). O(N*D*P + N*P^2) complexity. Supports both float and Q-format.
- **`fft1d.hpp`** — Radix-2 decimation-in-time FFT with compile-time bit-reversal tables and scaled butterfly stages. Twiddle factors injected externally for Q-format compatibility.
- **`batchnorm.hpp`** — Batch normalization with training/inference modes.
- **`dropout.hpp`** — Inverted dropout regularization.
- **`binarylayer.hpp`** — Binary neural network layer (XNOR+popcount).
- **`ternarylayer.hpp`** — Ternary neural network layer ({-1,0,+1} weights).

### Quantization (optional, `TINYMIND_ENABLE_QUANTIZATION=1`)

Post-training int8 quantization path that lives **alongside** the existing single-`ValueType` pipeline. None of the existing layers, `QValue`, or `NeuralNet<>` change; quantized models are built from a parallel layer family that templates on `<InputType, WeightType, AccumType, OutputType>` and carries TFLite/CMSIS-NN style runtime metadata (`scale`, `zero_point`, plus an integer `(multiplier, shift)` Requantizer between layers).

- **`qaffine.hpp`** — Affine primitives: `QAffineTensor`, `Requantizer<SrcAccum, DstStorage>`, the gemmlowp-style `saturatingRoundingDoublingHighMul` / `roundingDivideByPOT` helpers, and `multiplyByQuantizedMultiplier` (the int32-in / int32-out primitive used by QAdd and other multi-stage rescalers). Pure integer at runtime; freestanding-safe.
- **`qconv2d.hpp`**, **`qdepthwiseconv2d.hpp`** (per-channel weight scale, TFLite mandate), **`qpointwiseconv2d.hpp`**, **`qpool2d.hpp`** (`QMaxPool2D` / `QAvgPool2D` / `QGlobalAvgPool2D`), **`qdense.hpp`** — Quantized layer family. Weights, biases, and Requantizer tables are caller-owned; layer structs are pointer-shaped so the same model can be built once on the host and re-used across many MCU targets. Phase 10 adds `QConv2DPerChannel` and `QPointwiseConv2DPerChannel` siblings with the same template signature but `requantizers` arrays (one per output filter), matching the QDepthwiseConv2D pattern.
- **`qadd.hpp`**, **`qmul.hpp`**, **`qconcat.hpp`**, **`qpad.hpp`** — Phase 10 composition ops. `QAdd` uses TFLite ADD semantics (left_shift + 3 multiplier/shift triples); `QMul` is a single-Requantizer elementwise multiply; `QConcat2_2D` is channel-axis concat with per-input rescaler; `QPad2D` is constant pad using the input's zero_point so padded cells decode to true zero in the affine domain. All pure-integer at runtime.
- **`qbatchnorm.hpp`**, **`qlayernorm.hpp`**, **`qsoftmax.hpp`** — Phase 11 normalization ops. `QBatchNorm1D` / `QBatchNorm2D` apply a per-channel `(multiplier, shift, bias_addend)` triple (built host-side from float `gamma/beta/mean/variance` by `buildQBatchNormChannelParams`) for the standalone BN case that cannot be folded. `QLayerNorm1D` computes integer mean / variance per row, derives `1/sqrt(var)` via the pure-integer `qInvSqrtQ30` Newton iteration, then rescales by per-feature gamma (int16 Q1.14) and adds per-feature beta (int32 in output scale). `QSoftmax1D` is two-pass int8→int8 softmax: per-row max-subtract, 256-entry int32 exp LUT lookup (built host-side by `buildQSoftmaxExpLUT`), int64 sum, then normalize to the TFLite output convention (scale 1/256, zero_point -128). All pure-integer at runtime.
- **`qactivations.hpp`** — Quantized ReLU / ReLU6, plus `clampForRelu` / `clampForRelu6` helpers that fold the activation into the upstream Requantizer's saturation pass (matches CMSIS-NN runtime efficiency). 256-entry int8 lookup tables for sigmoid and tanh (`buildQSigmoidLUT` / `buildQTanhLUT` build host-side; `qApplyLUT` / `qApplyLUTBuffer` apply pure-integer at runtime).
- **`include/qcalibration.hpp`** — Host-only calibration helpers (`RangeObserver`, `computeAffineParamsAsymmetric` / `Symmetric`, `computePerChannelSymmetricScales`, `quantizeBuffer`, `buildRequantizer`). Phase 10 additions: `buildRescaler` (pure rescale, no weight component — for QConcat / QPad), `buildQAddParams` (TFLite ADD's three (multiplier, shift) triples around a fixed left_shift), `buildQMulRequantizer` (elementwise multiply). Phase 11 additions: `foldBatchNorm` (fold Conv2D + BatchNorm into one fused Conv2D pre-quantization), `buildQBatchNormChannelParams` (per-channel int32 triple for standalone `QBatchNorm`), `buildQSoftmaxExpLUT` (256-entry int32 exp table for `QSoftmax1D`). Gated on `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD` so the deployable inference binary never pulls in float math.

The deployable target shape is `TINYMIND_ENABLE_QUANTIZATION=1, TINYMIND_ENABLE_FLOAT=0, TINYMIND_ENABLE_STD=0`: int8 weights/activations, int32 accumulators, integer requantization, no `<cmath>`. The `unit_test/embedded` matrix exercises this corner as `quant_freestanding`. The `unit_test/quantization` Boost.Test suite covers the math (Requantizer round-trip, per-channel depthwise, calibration), and `examples/kws_cortex_m_int8/` is a side-by-side counterpart to `examples/kws_cortex_m/` with directly comparable CSV cycle/byte reports.

Phase 11 added `foldBatchNorm` (host-side Conv+BN fusion to fold BN into the upstream conv weights/biases pre-quantization), so the only-no-fusion-pass story now reads "no conv+bn+relu auto-fusion across the full graph"; BN can be folded explicitly at calibration time.

Non-goals in this phase: no QAT (post-training only), no int4/mixed precision, no whole-graph conv+bn+relu auto-fusion pass.

### Recurrent Quantization (optional, `TINYMIND_ENABLE_QUANTIZATION=1`)

Phase 12 adds int8 recurrent cells alongside the per-layer Q* family. They are standalone single-step cells; the caller owns the time loop and the hidden / cell state buffers.

- **`cpp/qlstm.hpp`** — `QLSTMCell<InputStorage, WeightStorage, AccumStorage, GateActStorage, HiddenStorage, CellStorage, NumInputs, NumHidden>`. Four gates (i, f, g, o) in TFLite ordering. Each gate carries two rescalers (input-MAC + recurrent-MAC) into a shared LUT input scale; pre-activation int32 is saturated to int8 then routed through the existing sigmoid / tanh LUTs (`qApplyLUT`). Cell update is `c_t = f_t * c_{t-1} + i_t * g_t` via two `multiplyByQuantizedMultiplier` calls; output is `o_t * tanh(c_t)` through a final `Requantizer`. `CellStorage = int8_t` is the deployable default; `CellStorage = int16_t` is the wide-cell variant for long unroll horizons (gate: `TINYMIND_ENABLE_INT16_ACCUM=1` advertises the int16 corner to the embedded matrix). Sigmoid / tanh LUT output conventions are TFLite-fixed (sigmoid 1/256 zp -128, tanh 1/128 zp 0); calibration helpers respect the same convention.
- **`cpp/qgru.hpp`** — `QGRUCell<InputStorage, WeightStorage, AccumStorage, GateActStorage, HiddenStorage, NumInputs, NumHidden>`. Three gates (r, z, n) in canonical ordering; reset-before-multiply formulation `n_t = tanh(W_n x + R_n (r_t * h_{t-1}) + b_n)`, hidden update `h_t = (1 - z_t) * n_t + z_t * h_{t-1}`. `(1 - z_t)` is computed exactly in the sigmoid grid as `-z_t` (real-domain identity confirmed at int8 precision).
- **`cpp/include/qcalibration.hpp`** additions: `QLSTMScales` / `QLSTMParams` / `buildQLSTMParams` plus `quantizeQLSTMBiases` decompose per-gate float scales into the (multiplier, shift) triples consumed by `QLSTMCell`; `QGRUScales` / `QGRUParams` / `buildQGRUParams` plus `quantizeQGRUBiases` do the same for `QGRUCell`. Host-only (gated on `FLOAT && STD`).

The `unit_test/embedded` matrix grows a sixth corner `int16_accum_freestanding` (`QUANT=1 INT16_ACCUM=1 FLOAT=0 STD=0`) that exercises both cells, including the int16-cell variant of `QLSTMCell`.

### Attention + FFT Quantization (optional, `TINYMIND_ENABLE_QUANTIZATION=1`)

Phase 13 adds int8 attention and FFT primitives alongside the per-layer Q* family. Same caller-owned-buffer / pure-integer-at-runtime pattern as the rest of `cpp/q*.hpp`.

- **`cpp/qfft1d.hpp`** — `QFFT1D<N>`. Radix-2 DIT FFT on int16 buffers with Q1.15 twiddle factors (caller-owned, built host-side by `buildQFFTTwiddles`). Scaled butterflies (right-shift by 1 each stage; total scaling 1/N) keep the int16 working register bounded. `magnitudeSquared` emits int32; the int8 boundary on either side is expressed as an ordinary `Requantizer`. Inverse via the conjugate trick (unscaled).
- **`cpp/qattention1d.hpp`** — `QAttention1D<InputStorage, WeightStorage, AccumStorage, ProjStorage, IntermStorage, OutputStorage, SequenceLength, EmbeddingDim, ProjectionDim>`. Linear (ReLU-kernel) self-attention, int8 counterpart of `selfattention1d.hpp`. Computes `Q' = qrelu(X W_q)`, `K' = qrelu(X W_k)`, `V = X W_v`, `KV = K'^T V`, `Y = Q' KV`. Each MAC has its own `Requantizer`; the ReLU on Q'/K' is folded by raising `qmin = zero_point`. Caller-owned weight, bias, and scratch buffers (Q', K', V', KV).
- **`cpp/qattention_softmax.hpp`** — `QAttentionSoftmax1D<...>`. Standard softmax-attention. Adds an `S x S` score buffer and an attention buffer (1/256 scale, zp = -128 per TFLite). Score requantizer folds the `1 / sqrt(d_k)` factor via the `qAttentionInvSqrt(P)` helper. Softmax exp LUT is the same 256-entry int32 table that `QSoftmax1D` uses (`buildQSoftmaxExpLUT`).
- **`cpp/qmha.hpp`** — `QMultiHeadLinearAttention1D<..., NumHeads>`. Holds `NumHeads` independent `QAttention1D` heads and stacks per-head outputs along the projection axis. Scratch buffers (Q', K', V', KV, per-head output) are reused across heads since they run sequentially.
- **`cpp/include/qcalibration.hpp`** additions: `buildQFFTTwiddles(n, cos_out, sin_out)` emits the Q1.15 sin/cos table; `QAttention1DScales` / `QAttentionSoftmaxScales` document the float scale conventions; `qAttentionInvSqrt(P)` returns `1/sqrt(P)` for folding the score-scaling factor. Host-only (gated on `FLOAT && STD`).

The `unit_test/embedded` matrix continues to exercise the freestanding (`quant_freestanding`) corner with the new headers, confirming they stay clear of `<cmath>` / `<type_traits>` / stdlib. `unit_test/quantization` adds five Phase 13 tests: Q1.15 twiddle round-trip, `QFFT1D` magnitude-spectrum parity vs a naive float DFT, `QFFT1D` forward/inverse round-trip, `QAttention1D` parity vs a float linear-attention reference, `QAttentionSoftmax1D` parity vs a float softmax-attention reference, and a `QMultiHeadLinearAttention1D` stacking test against the single-head ground truth. `examples/transformer_encoder_int8/` runs a single encoder block end-to-end (LayerNorm + QAttention1D + QAdd + LayerNorm + QDense + qrelu + QDense + QAdd) and reports max-abs error vs the float reference (~2% of output range on the bundled synthetic dataset).

### Mixed Precision Bridges (optional, `TINYMIND_ENABLE_FP16=1` and/or `TINYMIND_ENABLE_FLOAT=1`)

Phase 9 of the roadmap adds composability between the previously orphaned `QValue` (Q-format) and `QAffineTensor` (int8 affine) pipelines, plus a half-precision storage tier for application-class CPUs.

- **`cpp/qbridge.hpp`** — Pointwise type converters at layer boundaries: `affineDequantize` / `affineQuantize`, `qValueToFloat` / `floatToQValue`, `qValueToAffine` / `affineToQValue`, plus buffer-batch versions. Float at runtime, no `<cmath>` (rounding via sign-aware cast). Gated on `TINYMIND_ENABLE_FLOAT`; freestanding-safe at `STD=0`. Enables hybrid pipelines like *int8 affine CNN frontend → Q-format LSTM head → int8 affine classifier*. Phase 17 adds a parallel pure-integer path inside the same header gated on `TINYMIND_ENABLE_QUANTIZATION` (independent of `FLOAT`): `AffineToQValueIntParams<QV>` / `QValueToAffineIntParams<QV>` + `affineToQValueInt` / `qValueToAffineInt` (and buffer variants) reuse the gemmlowp Q0.31 `multiplyByQuantizedMultiplier` primitive, so the deployable freestanding shape `FLOAT=0 STD=0 QUANT=1` can mix Q-format and int8 affine tiers at runtime without `<cmath>`. Host-side helpers `buildAffineToQValueIntParams<QV>` / `buildQValueToAffineIntParams<QV>` build the integer triples at calibration time and ship them as data.
- **`cpp/include/tinymind_fp16.hpp`** — Software-only `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) storage structs wrapping `uint16_t`. Conversion helpers (`floatToFp16` / `fp16ToFloat`, `floatToBf16` / `bf16ToFloat`) handle normals, subnormals, Inf, and NaN. Storage tier; SIMD specializations land via Phase 14's `simd_neon_fp16.hpp` (NEON FEAT_FP16 vector forms).
- **`cpp/qbridge.hpp`** also provides `fp16ToAffineI8` / `affineI8ToFp16` / `bf16ToAffineI8` / `affineI8ToBf16` when `TINYMIND_ENABLE_FP16=1`.

The `unit_test/embedded` matrix exercises the float bridges as `fp16_freestanding` (`FLOAT=1 FP16=1 QUANT=1 STD=0`); the Phase 17 integer bridges ride in the `quant_freestanding` corner (`QUANT=1 FLOAT=0 STD=0`) so both halves stay freestanding-clean.

### SIMD Performance Backend (optional, `TINYMIND_ENABLE_SIMD_*=1`)

Phase 14 wires ISA-capability-gated SIMD specializations into the inner reduction loop of `QDense`, `QConv2D`, and `QConv2DPerChannel`. Every gate defaults to `0`; with all gates off the layer bodies fall back to a scalar dispatch that emits byte-identical output to the pre-Phase-14 build. The bench harness in `examples/perf_matrix/` confirms a `output_checksum` invariant across every enabled backend on the host.

- **Gates** (all default `0`): `TINYMIND_ENABLE_SIMD_NEON`, `TINYMIND_ENABLE_SIMD_NEON_DOTPROD`, `TINYMIND_ENABLE_SIMD_NEON_FP16`, `TINYMIND_ENABLE_SIMD_SVE`, `TINYMIND_ENABLE_SIMD_SVE2`, `TINYMIND_ENABLE_SIMD_HELIUM_MVE_I`, `TINYMIND_ENABLE_SIMD_HELIUM_MVE_F`, `TINYMIND_ENABLE_SIMD_AVX2`, `TINYMIND_ENABLE_SIMD_AVX_VNNI`, `TINYMIND_ENABLE_SIMD_AVX512F`, `TINYMIND_ENABLE_SIMD_AVX512_VNNI`, plus the orthogonal `TINYMIND_ENABLE_OPENMP`. Gates name ISA extensions, never CPU models — a Cortex-A55 configured without NEON simply does not set `SIMD_NEON=1`, no library change needed.
- **Prerequisite chain.** Each `simd_*.hpp` opens with a `static_assert` enforcing Arm's documented dependency table: `DOTPROD` requires `NEON`; `SVE`/`SVE2` require `NEON`; `FP16` (vector) requires `NEON`; `AVX_VNNI` requires `AVX2`; `AVX512_VNNI` requires `AVX512F`; the two Helium gates (`MVE_I`, `MVE_F`) are M-profile only and mutually exclusive with `NEON`/`SVE`. Misconfiguration like `DOTPROD=1, NEON=0` fails at compile time with a readable message; `unit_test/embedded/Makefile`'s `simd_prereq_regressions` target locks the regression by checking that those misconfigured builds fail.
- **`cpp/include/simd/`** — one header per capability. Each is self-contained: it does nothing unless its gate is on, then exposes a backend-namespaced `int8DotWithZeroPoint` primitive (e.g. `tinymind::simd::neon_dotprod::int8DotWithZeroPoint`). The public entry point lives in `cpp/include/simd/simd_dispatch.hpp` as `tinymind::simd::int8DotWithZeroPoint` (plus a templated `dotProductWithZeroPoint<Input, Weight, Accum>` that specializes on `int8_t/int8_t/int32_t`). Backend precedence on x86: `AVX512_VNNI > AVX512F > AVX_VNNI > AVX2 > scalar`; on Arm: `NEON_DOTPROD > NEON > SVE > HELIUM_MVE_I > scalar`. `activeBackendName()` exposes the resolved choice for benchmark reports.
- **`cpp/include/threading.hpp`** — `TINYMIND_PARALLEL_FOR_OUTER` macro that expands to `#pragma omp parallel for` when `TINYMIND_ENABLE_OPENMP=1` and nothing otherwise. Used on the output-filter loop of `QConv2D` / `QConv2DPerChannel`. Orthogonal to every SIMD gate; caller passes `-fopenmp` separately.
- **Bit-exactness.** Every integer backend is bit-exact with the scalar reference: int8 × int8 products fit in int16, the accumulation step preserves full int32 precision regardless of lane order, and the zero-point correction is folded into the final scalar subtract. The AVX2 backend avoids `PMADDUBSW` deliberately (it saturates on the pair-sum step); the AVX-VNNI and AVX-512-VNNI backends use the canonical uint8-shift trick so `VPDPBUSD` reduces a uint8/int8 product exactly. `unit_test/quantization/quantization_unit_test.cpp` adds five Phase 14 tests covering dispatch parity across pathological lengths, INT8 extreme values, and full `QDense` / `QConv2D` layers. Float SIMD reductions (`SIMD_NEON_FP16`, `SIMD_HELIUM_MVE_F`) are not bit-exact with scalar — the invariant applies only to the integer paths.
- **Embedded regression.** `unit_test/embedded/Makefile` adds the `simd_disabled` corner (every `SIMD_*=0`, `QUANT=1 FLOAT=0 STD=0`) to lock the scalar-fallback invariant at the deployable freestanding shape.
- **Bench.** `examples/perf_matrix/` builds the same int8 conv + dense block under each enabled gate combination and emits one CSV row per binary with `active_backend, conv_us_per_call, dense_us_per_call, output_checksum`. The checksum is identical across backends (bit-exactness gate); cycle counts vary by ISA.

Non-goals: no runtime CPU dispatch (`cpuid`, `getauxval`, `__builtin_cpu_supports`); library compiles for one ISA per build, fat-binary dispatch is the caller's problem. No `#ifdef __ARM_NEON` auto-detection in library headers; the build system translates `-march=` flags into matching `TINYMIND_ENABLE_SIMD_*=1` defines.

### Calibration Upgrades + Importer Tooling (Phase 15)

Phase 15 reduces the accuracy gap vs a PyTorch / TFLite reference and lowers friction to deploy. Three additions, all host-only.

- **`cpp/include/qcalibration.hpp` additions:**
  - `PercentileObserver` — records every sample, query `rangeAtPercentile(lower_pct, upper_pct, fmin_out, fmax_out)` for outlier-clipped affine bounds. Heavy-tail activations (post-softmax, large receptive-field conv) want this over naive min/max so the int8 grid is not wasted on a few extreme samples.
  - `KLDivergenceObserver` — TensorRT-style entropy calibration. Two passes: `observeAbsRange` to fix the 2048-bin histogram width, `observeHistogram` to fill it; `computeThreshold` sweeps candidate clip thresholds T in `[128, 2048]` and returns the one that minimizes KL between the reference distribution (clipped + tail-folded) and the int8-quantized distribution. Output is an absmax for symmetric quantization — feed into `computeAffineParamsSymmetric`.
  - `crossLayerEqualizeDense(w1, b1, w2, in, mid, out)` / `crossLayerEqualizeConv2D(w1, b1, w2, num_filters_1, weights_per_filter_1, num_filters_2, kh2, kw2)` — Nagel-paper Cross-Layer Equalization. For each intermediate channel `c`, compute `r1 = max|W1[c, :]|`, `r2 = max|W2[:, c]|`, `s = sqrt(r1 / r2)`, then `W1[c, :] /= s; b1[c] /= s; W2[:, c] *= s`. Output preserved under ReLU / identity (positively homogeneous). Pre-quantization pass; zero-row channels are skipped.
- **`apps/import_pytorch/tinymind_import.py`** — Python importer module. Caller assembles `Dense` / `Conv2D` / `BatchNorm2D` / `ReLU` / `Sigmoid` / `Tanh` / `Softmax` descriptors carrying numpy weights from `torch.state_dict`. `fuse_layers` folds Conv2D-then-BatchNorm pairs via the same math as `foldBatchNorm`. `calibrate` streams the calibration dataset through the float reference with `MinMaxObserver` / `PercentileObserver` / `KLDivergenceObserver` per layer. `quantize_weights` emits symmetric int8 weights + int32 biases. `emit_weights_header` writes a TinyMind-format `weights.hpp` whose shape matches the existing `examples/pytorch_quant/xor/weights.hpp`. Top-level `import_pytorch_model` wraps the four passes. No PyTorch dependency at module import — caller passes plain numpy plus a numpy forward callable.
- **`apps/import_onnx/tinymind_import_onnx.py`** — QDQ-format ONNX importer. Walks `QuantizeLinear` / `DequantizeLinear` / `QLinearConv` / `QLinearMatMul` / `Relu` / `Sigmoid` / `Tanh` / `Softmax` nodes from a model produced by `onnxruntime.quantization.quantize_static`, extracts the per-tensor `(scale, zero_point)` and int8 weight / int32 bias initializers, emits the same TinyMind-format `weights.hpp`. The `onnx` Python package is imported lazily inside `parse_onnx_model` so the emitter half is usable without it.

The Phase 15 deployable shape is unchanged: `TINYMIND_ENABLE_QUANTIZATION=1, FLOAT=0, STD=0`. All new helpers (Observers + CLE + importer scripts) live behind `FLOAT && STD` (the `qcalibration.hpp` gate) or in Python tooling. `examples/import_demo/` ships an end-to-end Phase 15 exemplar (C++ side: 3-8-4-2 MLP, three observers + CLE, calibration + int8 forward parity vs float, ~0.004 max-abs error on the bundled seed; Python side: full PyTorch-to-weights.hpp flow via `apps/import_pytorch`).

### Mixed-Precision Exemplars + Verification (Phase 16)

Phase 16 ships four reference int8 / mixed-precision exemplars and a `unit_test/integration` Boost.Test suite that locks their byte output across SIMD gate combos. Pure addition — no runtime header changes.

- **`examples/resnet18_block_int8/`** — int8 ResNet-18-shaped stem plus one basic-block stage (`QPad2D` → `QConv2DPerChannel 7x7 s=2` → `qrelu` → `QMaxPool2D` → basic block: `QPad2D` → 3x3 conv → `qrelu` → `QPad2D` → 3x3 conv → `QAdd` skip → `qrelu` → `QGlobalAvgPool2D` → `QDense`). Demonstrates that `QMaxPool2D`, `qreluBuffer`, and `QGlobalAvgPool2D` are pass-throughs on the int8 affine grid, so consecutive layers reuse the upstream `(scale, zero_point)` rather than burning new requantizers.
- **`examples/mobilenetv2_int8/`** — int8 MobileNetV2-shaped pipeline. Two inverted-residual blocks (one stride-1 with a residual skip, one stride-2 without), wired around a stride-2 stem and a GAP + dense head. The projection convolutions are linear (no `qrelu`), matching MNv2's "linear bottleneck" design rule. The inverted-residual unit is the load-bearing primitive of MNv2 / V3 / EfficientNet — the build pattern in this file scales linearly to a full model.
- **`examples/transformer_encoder_int8/`** — already present from Phase 13. Phase 16 wires it into the integration suite with the same `--golden` mode as the new exemplars.
- **`examples/mixed_precision_kws/`** — mixed-precision exemplar that exercises the Phase 9 qbridge converters in production shape: int8 `QDense` frontend → `affineI8ToFp16` bridge → fp16 linear-attention head with residual skip + mean-pool → `fp16ToAffineI8` bridge → int8 `QDense` classifier. `TINYMIND_ENABLE_FP16=1` required at the Makefile level. Inner attention arithmetic runs in float promoted from `fp16_t`; on targets that ship vector fp16 arithmetic (NEON FEAT_FP16, AVX-512 fp16) the promote pair is near-free, on every other target it is the cost of admission for fp16 storage on an MCU.

Each exemplar Makefile exposes the same three modes: `make run` (parity report vs float reference), `make bench` (CSV cycle/byte report), `make golden` (int8 byte stream for the bundled test set in a stable text format).

`unit_test/integration/` — new Boost.Test suite. One fixture per exemplar shells out to the example binary with `--golden` via `popen()` and compares the emitted byte stream to a baked-in expected string. The exemplar binaries are deterministic (hand-crafted weights, fixed synthetic dataset, pure-integer forward), so the output is invariant across SIMD gate combos by Phase 14's bit-exactness guarantee. Any silent drift in the example pipeline, the `qaffine.hpp` requantizer, the `qcalibration.hpp` helpers, or any SIMD specialization that claims bit-exactness trips the test. The root `Makefile`'s `check` target orders the integration test after the example builds so the binaries always exist when the test runs.

### Design Pattern

Neural networks are configured through a properties struct that bundles all template policies:

```cpp
// Example: define NN properties
struct XorNNProperties {
    typedef QValue<8, 8, false> ValueType;  // Q8.8 fixed-point
    static constexpr size_t NumberOfInputs = 2;
    static constexpr size_t NumberOfOutputs = 1;
    // ... layer sizes, learning rate, etc.
};
// Then instantiate:
typedef NeuralNet<XorNNProperties> XorNN;
```

### Unit Tests (`unit_test/`)

- **`nn/`** — Boost.Test unit tests for neural network correctness (training convergence, forward pass values, etc.)
- **`qformat/`** — Compile-time `static_assert` tests for fixed-point type properties
- **`qlearn/`** — Boost.Test unit tests for Q-learning
- **`quantization/`** — Boost.Test unit tests for the int8 quantization path: Requantizer round-trip, per-tensor / per-channel calibration, QConv2D / QDepthwise / QPointwise / QPool / QDense forward passes against a float reference. Phase 11 additions cover `foldBatchNorm` (fused-conv parity vs unfused conv→BN), `QBatchNorm2D` parity, `QLayerNorm1D` parity and constant-row edge case, and `QSoftmax1D` parity plus dominant-class saturation. Phase 12 additions cover `QLSTMCell` single-step parity vs a float LSTM reference, `QLSTMCell` int16-cell-state drift over a 256-step sequence, and `QGRUCell` single-step parity vs a float GRU reference. Phase 13 additions cover Q1.15 twiddle round-trip, `QFFT1D` magnitude-spectrum parity vs a naive float DFT, `QFFT1D` forward/inverse round-trip, `QAttention1D` parity vs a float linear-attention reference, `QAttentionSoftmax1D` parity vs a float softmax-attention reference, and a `QMultiHeadLinearAttention1D` stacking test. Phase 14 additions cover SIMD bit-exactness across pathological lengths, INT8 extreme-value patterns, full-layer `QDense` and `QConv2D` parity, and the `activeBackendName()` dispatch report. Phase 15 additions cover `PercentileObserver` outlier clipping + empty-buffer edge case, `KLDivergenceObserver` clip-threshold convergence vs a Gaussian + outliers dataset + empty edge case, `crossLayerEqualizeDense` output preservation under ReLU + zero-row skip, and `crossLayerEqualizeConv2D` output preservation. Builds with `TINYMIND_ENABLE_QUANTIZATION=1`; pass `-DTINYMIND_ENABLE_SIMD_*=1` plus the matching `-march=` flag to exercise a SIMD backend.
- **`embedded/`** — Cross-corner regression matrix. Builds the smoke source under eight `(FLOAT, STD, QUANT, FP16, INT16_ACCUM, SIMD_*)` configurations: `freestanding`, `no_stdlib`, `no_fpu`, `hosted`, `quant_freestanding`, `fp16_freestanding`, `int16_accum_freestanding`, and `simd_disabled` (Phase 14 scalar-fallback corner — every `TINYMIND_ENABLE_SIMD_*=0` at the deployable freestanding shape). A separate `simd_prereq_regressions` make target locks the static_assert prerequisite chain (`AVX_VNNI=1, AVX2=0` and `AVX512_VNNI=1, AVX512F=0` must fail to compile).
- **`integration/`** — Phase 16 golden-byte suite (extended in Phase 17). One Boost.Test fixture per exemplar (`resnet18_block_int8`, `mobilenetv2_int8`, `mixed_precision_kws`, `transformer_encoder_int8`, `mixed_precision_mlp_int8_qformat`) shells out to the example binary with `--golden` and asserts the emitted int8 byte stream matches a baked-in expected string. Catches silent regressions in the inference path regardless of which SIMD backend dispatch resolves to.

### Examples (`examples/`)

- **`xor/`** — XOR gate learned by a small neural network; includes a Python plotting script
- **`maze/`** and **`dqn_maze/`** — Maze solving via Q-learning and deep Q-networks
- **`pytorch/`** — Exports weights from a PyTorch model and imports them into a TinyMind C++ network for inference (Q-format pipeline)
- **`pytorch_quant/xor/`** — Affine int8 counterpart to `pytorch/xor/`. PyTorch float training + per-tensor calibration + `weights.hpp` emission, then a pure-integer C++ forward pass through `QDense` + `qrelu` + `QDense` + int8 sigmoid LUT
- **`kws_cortex_m/`** — Keyword-spotting-style pipeline built from `Conv2D` → `MaxPool2D` → `DepthwiseConv2D` → `PointwiseConv2D` → `GlobalAvgPool2D` → dense, with a CSV cycles/bytes report from the bench harness. Host runner; includes a `port_stub.hpp` sketch for porting to a Cortex-M target.
- **`kws_cortex_m_int8/`** — int8 quantized counterpart of `kws_cortex_m`. Same pipeline shape, same CSV report format, but every layer is replaced with the `cpp/q*.hpp` family. Demonstrates host-side calibration via `qcalibration.hpp` plus a pure-integer forward path. Use it for direct cycle/byte comparisons against the float pipeline.
- **`resnet_block_int8/`** — Phase 10 demonstration: int8 residual block (`QPad2D` → `QConv2DPerChannel` → `qrelu` → `QPad2D` → `QConv2DPerChannel` → `QAdd` with identity skip → `qrelu`). Calibrates per-channel weight scales and per-tensor activations on the host, then runs the block end-to-end on int8 and reports max-abs error vs the float reference.
- **`transformer_encoder_int8/`** — Phase 13 demonstration: int8 transformer encoder block (`QLayerNorm1D` → `QAttention1D` linear attention → `QAdd` skip → `QLayerNorm1D` → `QDense` + `qrelu` → `QDense` → `QAdd` skip). Calibrates per-tensor activations and per-tensor symmetric weight scales on the host, then runs the block end-to-end on int8 and reports max-abs error vs the float reference (~2% of output range on the bundled dataset).
- **`perf_matrix/`** — Phase 14 SIMD gate bench. Builds the same `QConv2D` 3x3 + `QDense` int8 block under each enabled `TINYMIND_ENABLE_SIMD_*` combination on the host (default Makefile builds scalar / AVX2 / AVX-512F / AVX-512-VNNI). Emits one CSV row per backend with per-call timing and an `output_checksum` that is invariant across backends — the row is the bit-exactness regression and the cycle delta is the perf headline.
- **`import_demo/`** — Phase 15 importer end-to-end. C++ binary carries a deterministic 3-8-4-2 MLP, drives a 64-sample synthetic calibration set through `RangeObserver` / `PercentileObserver` / `KLDivergenceObserver` plus `crossLayerEqualizeDense`, then runs both the float reference and the pure-integer int8 forward and reports max-abs error (~0.004 on the bundled seed; tolerance 0.08). Standalone — no torch dependency. `demo.py` is the production-flow counterpart that consumes `torch.state_dict` and drives `apps/import_pytorch/tinymind_import` to emit a real `weights.hpp`.
- **`resnet18_block_int8/`** — Phase 16 exemplar. int8 ResNet-18-shaped stem + one basic-block stage on a 16x16x3 input, 4 logits out. Same `make run` / `make bench` / `make golden` mode triple as the other Phase 16 exemplars.
- **`mobilenetv2_int8/`** — Phase 16 exemplar. int8 MobileNetV2-shaped pipeline: stride-2 stem + one stride-1 inverted-residual block with skip + one stride-2 inverted-residual block, then GAP + dense. Linear bottlenecks per MNv2 convention.
- **`mixed_precision_kws/`** — Phase 16 mixed-precision exemplar. int8 `QDense` frontend → Phase 9 `affineI8ToFp16` bridge → fp16 linear-attention head with residual skip + mean-pool → Phase 9 `fp16ToAffineI8` bridge → int8 `QDense` classifier. Requires `TINYMIND_ENABLE_FP16=1`.
- **`mixed_precision_mlp_int8_qformat/`** — Phase 17 hybrid mixed-precision exemplar. int8 `QDense` frontend → `qrelu` → Phase 17 `affineToQValueIntBuffer` (pure-integer bridge) → Q8.8 dense matvec (int32 accumulator) → Phase 17 `qValueToAffineIntBuffer` (pure-integer bridge) → int8 `QDense` classifier. Deployable shape is `QUANT=1 FLOAT=0 STD=0`; the exemplar builds hosted for the parity report (~0.005 max-abs error vs the float reference) and wires into the integration suite via the same `--golden` mode.

### Apps (`apps/`)

- **`apps/activation/`** — Standalone tool that generates the lookup table values used in `lookupTables.cpp`.
- **`apps/import_pytorch/`** — Phase 15 PyTorch → TinyMind int8 importer module (`tinymind_import.py`). Pure-numpy core: caller assembles a layer descriptor list from a `torch.state_dict`, picks a per-layer observer (`MinMaxObserver` / `PercentileObserver` / `KLDivergenceObserver`), calls `import_pytorch_model(...)`, gets a TinyMind-format `weights.hpp`. Conv2D-then-BatchNorm2D pairs auto-fuse via `fuse_layers` ahead of calibration.
- **`apps/import_onnx/`** — Phase 15 QDQ-format ONNX → TinyMind int8 importer (`tinymind_import_onnx.py`). Targets canonical CNN classifier topology: `QLinearConv` / `QLinearMatMul` plus `Relu` / `Sigmoid` / `Tanh` / `Softmax`. The `onnx` package is imported lazily so the emitter half is usable without it.

### Benchmark harness (`cpp/include/bench/`)

- **`platform.hpp`** — `bench::readCycleCounter()` (DWT CYCCNT on Cortex-M, `std::chrono` host fallback) and `bench::paintStack` / `bench::stackHighWater` for MCU stack watermarking. Enable the MCU path with `-DTINYMIND_BENCH_CORTEX_M`.
- **`report.hpp`** — `bench::LayerStat` CSV row type, `bench::writeHeader` / `bench::writeRow` for any sink that supports `operator<<`, and a `bench::ScopedTimer` for per-layer measurements. See `examples/kws_cortex_m/` for an end-to-end use.
