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

Eight preprocessor macros in `cpp/include/tinymind_platform.hpp` control optional hosted dependencies. **All default to 0** (freestanding); hosted Makefiles set `=1` selectively.

- `TINYMIND_ENABLE_FLOAT` — `float`/`double` as `ValueType`
- `TINYMIND_ENABLE_STD` — `<cmath>`, `<type_traits>`, `namespace std::`
- `TINYMIND_ENABLE_HOSTED_IO` — `<fstream>`, `<vector>`, `<cstdlib>`, `<cstdio>` (file weight serialization)
- `TINYMIND_ENABLE_OSTREAMS` — `<ostream>` and `QValue::operator<<`
- `TINYMIND_ENABLE_HOSTED_RAND` — `<cstdlib>` `rand()` (Dropout training mode, ScheduledSampling, Xavier)
- `TINYMIND_ENABLE_QUANTIZATION` — `cpp/q*.hpp` int8 quantization layer family + `Requantizer`. Calibration helpers in `cpp/include/qcalibration.hpp` are additionally gated on `FLOAT && STD` (host-only). Existing fixed-point and float pipelines are unaffected.
- `TINYMIND_ENABLE_FP16` — Phase 9 half-precision storage tier: `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) in `cpp/include/tinymind_fp16.hpp`, with float conversion helpers. Storage only — SIMD specializations land in Phase 14. Conversion helpers additionally require `TINYMIND_ENABLE_FLOAT`.
- `TINYMIND_ENABLE_INT16_ACCUM` — Phase 12 wide carry-state for `QLSTMCell` (cell-state stored as `int16_t` rather than `int8_t`). The cell template itself only needs the C++ type, so the gate purely controls whether the embedded regression matrix advertises the int16 corner; targets without an int16 storage path simply do not instantiate the wide template.

`unit_test/embedded` is the regression guard — its Makefile builds seven corners of the `(FLOAT, STD, QUANT, FP16, INT16_ACCUM)` matrix (`freestanding`, `no_stdlib`, `no_fpu`, `hosted`, `quant_freestanding`, `fp16_freestanding`, `int16_accum_freestanding`) and `make check` runs all of them. Float-typed Adam/RMSprop/Xavier require both `FLOAT && STD`. When adding a new header to `cpp/`, include `"include/tinymind_platform.hpp"` first and gate any `<cmath>`/`<type_traits>`/`std::` use; for SFINAE on `is_floating_point`, use `tinymind::enable_if` and `tinymind::is_floating_point` from `cpp/include/tinymind_traits.hpp` so the layer compiles at `STD=0`.

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

- **`cpp/qbridge.hpp`** — Pointwise type converters at layer boundaries: `affineDequantize` / `affineQuantize`, `qValueToFloat` / `floatToQValue`, `qValueToAffine` / `affineToQValue`, plus buffer-batch versions. Float at runtime, no `<cmath>` (rounding via sign-aware cast). Gated on `TINYMIND_ENABLE_FLOAT`; freestanding-safe at `STD=0`. Enables hybrid pipelines like *int8 affine CNN frontend → Q-format LSTM head → int8 affine classifier*.
- **`cpp/include/tinymind_fp16.hpp`** — Software-only `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) storage structs wrapping `uint16_t`. Conversion helpers (`floatToFp16` / `fp16ToFloat`, `floatToBf16` / `bf16ToFloat`) handle normals, subnormals, Inf, and NaN. Storage tier only; SIMD specializations (NEON fp16, AVX-512 fp16) land in Phase 14.
- **`cpp/qbridge.hpp`** also provides `fp16ToAffineI8` / `affineI8ToFp16` / `bf16ToAffineI8` / `affineI8ToBf16` when `TINYMIND_ENABLE_FP16=1`.

The `unit_test/embedded` matrix exercises this as `fp16_freestanding` (`FLOAT=1 FP16=1 QUANT=1 STD=0`) to confirm the half-precision and bridge headers stay freestanding-clean.

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
- **`quantization/`** — Boost.Test unit tests for the int8 quantization path: Requantizer round-trip, per-tensor / per-channel calibration, QConv2D / QDepthwise / QPointwise / QPool / QDense forward passes against a float reference. Phase 11 additions cover `foldBatchNorm` (fused-conv parity vs unfused conv→BN), `QBatchNorm2D` parity, `QLayerNorm1D` parity and constant-row edge case, and `QSoftmax1D` parity plus dominant-class saturation. Phase 12 additions cover `QLSTMCell` single-step parity vs a float LSTM reference, `QLSTMCell` int16-cell-state drift over a 256-step sequence, and `QGRUCell` single-step parity vs a float GRU reference. Phase 13 additions cover Q1.15 twiddle round-trip, `QFFT1D` magnitude-spectrum parity vs a naive float DFT, `QFFT1D` forward/inverse round-trip, `QAttention1D` parity vs a float linear-attention reference, `QAttentionSoftmax1D` parity vs a float softmax-attention reference, and a `QMultiHeadLinearAttention1D` stacking test. Builds with `TINYMIND_ENABLE_QUANTIZATION=1`.
- **`embedded/`** — Cross-corner regression matrix. Builds the smoke source under seven `(FLOAT, STD, QUANT, FP16, INT16_ACCUM)` configurations including `quant_freestanding` (`FLOAT=0 STD=0 QUANT=1`) which is the deployable inference shape for an int8 MCU target and `int16_accum_freestanding` which exercises the Phase 12 wide-cell QLSTM corner.

### Examples (`examples/`)

- **`xor/`** — XOR gate learned by a small neural network; includes a Python plotting script
- **`maze/`** and **`dqn_maze/`** — Maze solving via Q-learning and deep Q-networks
- **`pytorch/`** — Exports weights from a PyTorch model and imports them into a TinyMind C++ network for inference (Q-format pipeline)
- **`pytorch_quant/xor/`** — Affine int8 counterpart to `pytorch/xor/`. PyTorch float training + per-tensor calibration + `weights.hpp` emission, then a pure-integer C++ forward pass through `QDense` + `qrelu` + `QDense` + int8 sigmoid LUT
- **`kws_cortex_m/`** — Keyword-spotting-style pipeline built from `Conv2D` → `MaxPool2D` → `DepthwiseConv2D` → `PointwiseConv2D` → `GlobalAvgPool2D` → dense, with a CSV cycles/bytes report from the bench harness. Host runner; includes a `port_stub.hpp` sketch for porting to a Cortex-M target.
- **`kws_cortex_m_int8/`** — int8 quantized counterpart of `kws_cortex_m`. Same pipeline shape, same CSV report format, but every layer is replaced with the `cpp/q*.hpp` family. Demonstrates host-side calibration via `qcalibration.hpp` plus a pure-integer forward path. Use it for direct cycle/byte comparisons against the float pipeline.
- **`resnet_block_int8/`** — Phase 10 demonstration: int8 residual block (`QPad2D` → `QConv2DPerChannel` → `qrelu` → `QPad2D` → `QConv2DPerChannel` → `QAdd` with identity skip → `qrelu`). Calibrates per-channel weight scales and per-tensor activations on the host, then runs the block end-to-end on int8 and reports max-abs error vs the float reference.
- **`transformer_encoder_int8/`** — Phase 13 demonstration: int8 transformer encoder block (`QLayerNorm1D` → `QAttention1D` linear attention → `QAdd` skip → `QLayerNorm1D` → `QDense` + `qrelu` → `QDense` → `QAdd` skip). Calibrates per-tensor activations and per-tensor symmetric weight scales on the host, then runs the block end-to-end on int8 and reports max-abs error vs the float reference (~2% of output range on the bundled dataset).

### Apps (`apps/activation/`)

Standalone tool that generates the lookup table values used in `lookupTables.cpp`.

### Benchmark harness (`cpp/include/bench/`)

- **`platform.hpp`** — `bench::readCycleCounter()` (DWT CYCCNT on Cortex-M, `std::chrono` host fallback) and `bench::paintStack` / `bench::stackHighWater` for MCU stack watermarking. Enable the MCU path with `-DTINYMIND_BENCH_CORTEX_M`.
- **`report.hpp`** — `bench::LayerStat` CSV row type, `bench::writeHeader` / `bench::writeRow` for any sink that supports `operator<<`, and a `bench::ScopedTimer` for per-layer measurements. See `examples/kws_cortex_m/` for an end-to-end use.
