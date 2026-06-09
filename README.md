# TinyMind

[![Static & Dynamic Analysis](https://github.com/danmcleran/tinymind/actions/workflows/analysis.yml/badge.svg)](https://github.com/danmcleran/tinymind/actions/workflows/analysis.yml)
[![CodeQL](https://github.com/danmcleran/tinymind/actions/workflows/codeql.yml/badge.svg)](https://github.com/danmcleran/tinymind/actions/workflows/codeql.yml)

A header-only C++ template library for neural networks, Kolmogorov-Arnold Networks (KAN), LSTM and GRU recurrent networks, liquid neural networks (LTC and CfC continuous-time cells), linear self-attention, FFT-based signal processing, 1D and 2D convolutions (including MobileNet-style depthwise-separable blocks), binary and ternary neural networks, and Q-learning, designed for embedded systems with no FPU, GPU, or vectorized instruction requirements.

Inspired by Andrei Alexandrescu's policy-based design from [Modern C++ Design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design), TinyMind uses template metaprogramming to produce zero-overhead abstractions where network topology, value type, activation functions, and training policies are all compile-time parameters.

## Features

### Neural Networks

- **Feed-forward networks** with arbitrary depth and width
- **1D convolution layer** for time-series feature extraction (sensor data, IMU, ECG)
- **1D pooling layers** (`MaxPool1D`, `AvgPool1D`) for downsampling with multi-channel support and backpropagation
- **2D convolution layer** (`Conv2D`) with NHWC layout (channel-last) for spectrograms, images, and time-frequency tiles -- MFCC/keyword-spotting and small vision workloads
- **Depthwise-separable blocks** (`DepthwiseConv2D` + `PointwiseConv2D`) -- MobileNet-style ~8-9x MAC reduction vs. full 2D convolution at K=3
- **2D pooling layers** (`MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`) with backpropagation -- GAP replaces the flatten-to-dense matrix that dominates flash in small CNNs
- **Linear self-attention** (`SelfAttention1D`) using ReLU kernel feature map -- O(N*P^2) instead of O(N^2*D), no softmax/exp required, works with Q-format fixed-point
- **FFT layer** (`FFT1D`) with radix-2 decimation-in-time, compile-time bit-reversal tables, and scaled butterfly stages for fixed-point overflow prevention -- frequency-domain feature extraction for signal processing pipelines
- **Kolmogorov-Arnold Networks (KAN)** with learnable B-spline activation functions on edges
- **Recurrent neural networks** (Elman) with configurable recurrent connection depth
- **LSTM networks** with gated cell state, supporting single and multi-layer configurations
- **GRU networks** (Gated Recurrent Unit) with 3-gate architecture -- ~25% less memory than LSTM per hidden neuron
- **Binary neural networks** (`BinaryDense`) with XNOR+popcount forward pass -- 32x weight memory reduction via bit packing, no multiplication needed
- **Ternary neural networks** (`TernaryDense`) with {-1, 0, +1} weights -- 16x weight memory reduction via 2-bit packing, multiply-free forward pass using conditional add/subtract/skip
- **Heterogeneous hidden layers** via `HiddenLayers<N0, N1, ...>` for different neuron counts per layer (`MultilayerPerceptron` is a convenience alias for uniform layers)
- **Batch training** with configurable batch size
- **Softmax output** for multi-class classification
- **Xavier weight initialization** (uniform and normal distributions)
- **Weight import/export** in CSV and binary formats for all network types (MLP, LSTM, GRU, KAN; interoperable with PyTorch)

### Training Policies

- **Adam optimizer** with per-parameter adaptive learning rates (`AdamOptimizerFloat` for double, `AdamOptimizer` for fixed-point)
- **RMSprop optimizer** with per-parameter adaptive learning rates via running average of squared gradients (`RmsPropOptimizerFloat` for double, `RmsPropOptimizer` for fixed-point) -- often preferred over Adam for recurrent networks
- **Dropout regularization** via `Dropout<ValueType, Size, DropoutPercent>` -- inverted dropout with training/inference mode toggle, no scaling needed at inference time
- **Gradient clipping** via configurable `GradientClipByValue` policy to prevent exploding gradients (especially critical for fixed-point)
- **L2 weight decay** (ridge regularization) via `L2WeightDecay` policy to prevent weight overflow and reduce overfitting
- **Learning rate scheduling** with `StepDecaySchedule` (multiply by decay factor every N steps) and `FixedLearningRatePolicy` (default)
- **Early stopping** via `EarlyStopping<ValueType, Patience>` to detect convergence and save compute cycles
- **Teacher forcing / scheduled sampling** via `ScheduledSampling` for recurrent training -- linearly decays from ground truth to model predictions
- **Truncated BPTT** via `TruncatedBPTT<NNType, WindowSize>` -- accumulates recurrent state over K timesteps before weight update
- All policies are optional template parameters with null/no-op defaults for full backward compatibility
- Policies are extracted from `TransferFunctionsPolicy` via SFINAE traits -- existing user code compiles unchanged

### Kolmogorov-Arnold Networks (KAN)

- Based on the [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)
- Learnable B-spline activation functions on edges, pure summation at nodes
- Edge function: `phi(x) = w_b * SiLU(x) + w_s * spline(x)`
- Configurable spline degree (`SplineDegree`) and grid resolution (`GridSize`)
- Piecewise linear specialization (`SplineDegree=1`) for fixed-point targets
- SiLU activation reuses existing sigmoid lookup tables -- no new tables needed
- Supports both training and inference-only modes via `IsTrainable` template parameter
- Same user-facing API as `MultilayerPerceptron`: `feedForward`, `trainNetwork`, `calculateError`, `getLearnedValues`

### Forward-Mode Autodiff (Physics-Informed Neural Networks)

- `Dual<ValueType>` (`cpp/dual.hpp`) -- forward-mode automatic-differentiation dual number, carrying a value and its derivative with respect to one seeded input direction
- Built purely from the value type's `+`, `-`, `*`, `/`, so it works identically for `float`/`double` and fixed-point `QValue`; freestanding-clean (no `<cmath>`, STD, or FLOAT required -- runs in the deployable MCU build)
- Elementary functions over duals: `tanh` / `sigmoid` (`cpp/dualActivations.hpp`) and `exp` / `sin` / `cos` / `sqrt` (`cpp/dualmath.hpp`), each with analytic derivatives -- enough for SIREN-style fields and PDEs with exp/trig source terms (`sin`/`cos`/`exp` for `float`/`double`; `sqrt` for all types incl. fixed-point)
- Nested `Dual<Dual<...>>` yields 2nd- and higher-order derivatives, including mixed partials (`d²u/dx dy`) via per-level seed directions
- Provides the input-coordinate derivatives (`du/dx`, `d^2u/dx^2`) a PDE residual needs -- the basis for [Physics-Informed Neural Networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks). Primary deployment is train-offline / inference-only (a plain forward pass of `u(x, t)`) in `double` (exact) or Q-format (quantized)
- Reusable training machinery (`cpp/pinn.hpp`): `PinnMlp` is a Dual-differentiable MLP (differentiates w.r.t. inputs *and* weights, and runs plain inference); `sgdStep` is a PDE-agnostic momentum-SGD step that gets the **exact** loss gradient w.r.t. all weights in one pass via vector forward-mode (`MultiDual<V,N>`, `cpp/multidual.hpp`) -- no finite-difference error
- `tinymind::pinn::forwardAs` makes a **stock** trained `NeuralNetwork` (feed-forward, uniform-width hidden layers) input-differentiable: re-evaluates it in any scalar type via the public weight getters, so `Dual` inputs give `du/dx` -- purely additive, the network's own forward/train path is untouched
- Higher-order ergonomics: `Jet<V,Order>` (`cpp/taylor.hpp`) gives arbitrary-order single-variable derivatives (`u_xxx`...) in one sweep; `RevVar` (`cpp/revdual.hpp`) is a tape-based reverse-mode scalar that computes the loss gradient w.r.t. all weights in one backward pass (composes under `Dual` for reverse-over-forward; host-only), with `pinn::sgdStepReverse` as the reverse-mode trainer step
- Host PINN training is ordinary host code (no device needed): [`examples/pinn_heat1d/`](examples/pinn_heat1d/) `make train` fits the 1-D heat equation `u_t - nu*u_xx`, driving the PINN loss down ~50x to ~2.5% solution error, then benchmarks the residual in Q16.16 fixed point against the `double` residual (~6e-3)
- Unit-tested end to end in [`unit_test/pinn/`](unit_test/pinn/): `PinnMlp` forward / input-derivative / weight-gradient parity, `sgdStep` vs `sgdStepReverse` agreement, and an automated heat-equation residual convergence gate (the pass/fail counterpart of the example) -- on top of the scalar autodiff coverage in [`unit_test/dual/`](unit_test/dual/)
- See [`docs/pinn-feasibility.md`](docs/pinn-feasibility.md) for the full feasibility analysis and deployment paths

### Liquid Neural Networks (Continuous-Time)

Continuous-time recurrent cells from the MIT liquid-network line of work. Both are standalone single-step cells (`step<S>` / `forward`); the caller owns the state buffer and the time loop, matching the `QLSTMCell` / `QGRUCell` convention. The float cells are written scalar-templated in the PINN style, so they **train through the existing autodiff machinery** (`MultiDual` / `RevVar` + `pinn::sgdStep` / `sgdStepReverse`) with no hand-written backprop.

- **Liquid Time-Constant (LTC)** -- `LtcCell<NumInputs, NumState, Act>` (`cpp/ltc.hpp`), the fused (semi-implicit Euler) ODE solver from Hasani et al., *Liquid Time-constant Networks* (AAAI 2021). Per-neuron `dx/dt = -[1/tau + f]*x + f*A`; the fused step `x(t+dt) = (x + dt*f*A) / (1 + dt*(1/tau + f))` is unconditionally stable (denominator > 1) with no inner iteration. `step<S>` infers in `double`/`float`/`QValue` and differentiates through `Dual`/`MultiDual`/`RevVar` (reverse-mode training behind `-DTINYMIND_LTC_REVERSE_TRAINING=1`). The float / fixed-point liquid tier
- **Closed-form Continuous-time (CfC)** -- `CfCCell<NumInputs, NumState, BackboneDim, ...>` (`cpp/cfc.hpp`), the solver-free sibling from Hasani et al., *Closed-form continuous-time neural networks* (Nature MI 2022). Backbone trunk over `[input ++ h_prev]`, two tanh heads, a time-gate `t = sigmoid(tA*ts + tB)`, and the interpolation `h' = (1-t)*ff1 + t*ff2`. The per-step elapsed time `ts` is a runtime scalar, so **irregular sampling** is supported directly. Same scalar-templated autodiff story as LTC (training behind `-DTINYMIND_CFC_REVERSE_TRAINING=1`)
- **int8 deployable CfC** -- `QCfCCell<...>` (`cpp/qcfc.hpp`), the pure-integer counterpart (regular-sampling form: `ts` folded into the time-A requantizer at calibration). Each MAC carries its own `Requantizer` into the shared sigmoid/tanh LUT input scale; freestanding-safe under `FLOAT=0 STD=0`. Host calibration via `QCfCScales` / `buildQCfCParams` / `quantizeQCfCBias` / `quantizeQCfCTimeBias`. (LTC's continuous `1/tau` dynamics deliberately stay float / fixed-point; CfC is the int8 tier)
- Demos: [`examples/ltc_sequence/`](examples/ltc_sequence/) (LTC trained to a leaky-integrator step response, ~100x loss reduction) and [`examples/cfc_sequence/`](examples/cfc_sequence/) (CfC trained on an irregularly-sampled target, varying `ts` into the time-gate), both through `pinn::sgdStepReverse`; [`examples/qcfc_liquid_int8/`](examples/qcfc_liquid_int8/) is the int8 deployment exemplar (host calibration → pure-integer `QCfCCell` forward, `make run`/`bench`/`golden`)

### Fixed-Point Arithmetic

- `QValue<IntegerBits, FractionalBits, IsSigned>` template supporting Q8.8, Q16.16, Q24.8, Q32.32, and other formats up to 128-bit
- Full operator overloading (`+`, `-`, `*`, `/`, comparisons)
- Configurable rounding (`TruncatePolicy`, `RoundUpPolicy`) and saturation (`WrapPolicy`, `MinMaxSaturatePolicy`) policies
- Pre-computed lookup tables for sigmoid, tanh, exp, and log across all supported bit-widths
- Also supports `float` and `double` as value types for prototyping

### Post-Training Int8 Quantization (optional, `TINYMIND_ENABLE_QUANTIZATION=1`)

A parallel TFLite/CMSIS-NN style affine quantization path that runs **alongside** the existing single-`ValueType` pipeline (no changes to `QValue`, `NeuralNet<>`, or any current layer):

- **Convolution / dense family** -- `QConv2D`, `QConv2DPerChannel`, `QDepthwiseConv2D` (per-channel weight scale, TFLite mandate), `QPointwiseConv2D`, `QPointwiseConv2DPerChannel`, `QMaxPool2D`, `QAvgPool2D`, `QGlobalAvgPool2D`, `QDense` -- int8 weights/activations, int32 accumulators, integer requantization between layers via gemmlowp-style `Requantizer<int32, int8>` (Q0.31 multiplier + shift)
- **Composition ops** -- `QAdd` (TFLite ADD semantics, left_shift + 3 multiplier/shift triples), `QMul` (single-Requantizer elementwise multiply), `QConcat2_2D` (channel-axis concat with per-input rescaler), `QPad2D` (constant pad using input zero_point so padded cells decode to true zero)
- **Normalization** -- `QBatchNorm1D` / `QBatchNorm2D` (per-channel (multiplier, shift, bias) triple), `QLayerNorm1D` (integer mean/variance per row, `qInvSqrtQ30` Newton iteration), `QSoftmax1D` (two-pass int8->int8 softmax via 256-entry int32 exp LUT, output 1/256, zp -128)
- **Recurrent cells** -- `QLSTMCell` (4 gates, TFLite ordering, int8 or int16 cell state via `TINYMIND_ENABLE_INT16_ACCUM=1`), `QGRUCell` (3 gates, reset-before-multiply), and `QCfCCell` (closed-form continuous-time: backbone trunk + tanh heads + folded-`ts` time-gate + interpolation). Standalone single-step; caller owns time loop and hidden/cell buffers
- **Attention + FFT** -- `QFFT1D` (radix-2 DIT on int16 buffers, Q1.15 twiddles, scaled butterflies), `QAttention1D` (int8 linear-attention with ReLU kernel), `QAttentionSoftmax1D` (standard softmax attention with `1/sqrt(d_k)` folded into score Requantizer), `QMultiHeadLinearAttention1D` (stacks N heads)
- **Activations** -- `qrelu` / `qrelu6` plus `clampForRelu` / `clampForRelu6` helpers that fold the activation into the upstream Requantizer's saturation pass; 256-entry int8 sigmoid / tanh LUTs via `buildQSigmoidLUT` / `buildQTanhLUT` + `qApplyLUT` / `qApplyLUTBuffer` -- no `<cmath>` on the inference path
- **Host-side calibration** (`cpp/include/qcalibration.hpp`, gated on `FLOAT && STD`) -- `RangeObserver`, `PercentileObserver`, `KLDivergenceObserver` (TensorRT entropy), `crossLayerEqualizeDense` / `crossLayerEqualizeConv2D` (Nagel CLE), `computeAffineParamsAsymmetric` / `Symmetric`, `computePerChannelSymmetricScales`, `quantizeBuffer`, `buildRequantizer`, `buildRescaler`, `buildQAddParams`, `buildQMulRequantizer`, `foldBatchNorm` (Conv2D+BN fusion), `buildQBatchNormChannelParams`, `buildQSoftmaxExpLUT`, `buildQLSTMParams` / `quantizeQLSTMBiases`, `buildQGRUParams` / `quantizeQGRUBiases`, `buildQCfCParams` / `quantizeQCfCBias` / `quantizeQCfCTimeBias`, `buildQFFTTwiddles`, `qAttentionInvSqrt`
- **Pure integer at runtime**: deployable shape is `TINYMIND_ENABLE_QUANTIZATION=1, FLOAT=0, STD=0`; `unit_test/embedded` exercises this corner as `quant_freestanding`; `unit_test/quantization` Boost.Test suite covers the math (Requantizer round-trip, per-channel depthwise, calibration, Phase 11-15 ops, SIMD bit-exactness)
- **End-to-end examples**:
  - [`examples/pytorch_quant/xor/`](examples/pytorch_quant/xor/) -- PyTorch float training + per-tensor calibration + `weights.hpp` emission, then a pure-integer C++ forward pass through `QDense` + `qrelu` + `QDense` + int8 sigmoid LUT
  - [`examples/kws_cortex_m_int8/`](examples/kws_cortex_m_int8/) -- side-by-side counterpart to `examples/kws_cortex_m/`; same MobileNet-style KWS pipeline, comparable CSV cycle/byte report, ~4x smaller weight footprint on the convolutional layers
  - [`examples/resnet_block_int8/`](examples/resnet_block_int8/) -- int8 residual block (`QPad2D` -> `QConv2DPerChannel` -> `qrelu` -> `QPad2D` -> `QConv2DPerChannel` -> `QAdd` -> `qrelu`) with per-channel weight scales
  - [`examples/resnet18_block_int8/`](examples/resnet18_block_int8/) -- int8 ResNet-18-shaped stem + one basic-block stage. `make run`, `make bench`, `make golden`
  - [`examples/mobilenetv2_int8/`](examples/mobilenetv2_int8/) -- int8 MobileNetV2 inverted-residual block sequence with linear bottlenecks
  - [`examples/transformer_encoder_int8/`](examples/transformer_encoder_int8/) -- int8 encoder block: `QLayerNorm1D` -> `QAttention1D` -> `QAdd` -> `QLayerNorm1D` -> `QDense` + `qrelu` -> `QDense` -> `QAdd`. ~2% max-abs error vs float on bundled dataset
  - [`examples/mixed_precision_kws/`](examples/mixed_precision_kws/) -- mixed-precision: int8 frontend -> fp16 attention head -> int8 classifier. Exercises Phase 9 qbridge converters
  - [`examples/mixed_precision_mlp_int8_qformat/`](examples/mixed_precision_mlp_int8_qformat/) -- hybrid int8 affine <-> Q8.8 via Phase 17 pure-integer bridges. Deployable at `QUANT=1 FLOAT=0 STD=0`
  - [`examples/import_demo/`](examples/import_demo/) -- end-to-end Phase 15 importer flow. 3-8-4-2 MLP, three observers + CLE, ~0.004 max-abs error vs float
  - [`examples/perf_matrix/`](examples/perf_matrix/) -- SIMD gate bench. Same int8 conv + dense block under each enabled `TINYMIND_ENABLE_SIMD_*` combo, emits CSV per backend with invariant `output_checksum`
  - [`unit_test/integration/`](unit_test/integration/) -- Phase 16/17 golden-byte regression suite. Locks five exemplars' int8 output byte-for-byte across SIMD gate combos

### Mixed-Precision Bridges (optional, `TINYMIND_ENABLE_FP16=1` and/or `TINYMIND_ENABLE_FLOAT=1`)

Composability between the previously orphaned `QValue` (Q-format) and `QAffineTensor` (int8 affine) pipelines, plus a half-precision storage tier:

- **`cpp/qbridge.hpp`** -- pointwise type converters at layer boundaries: `affineDequantize` / `affineQuantize`, `qValueToFloat` / `floatToQValue`, `qValueToAffine` / `affineToQValue`, buffer-batch versions. Float path gated on `TINYMIND_ENABLE_FLOAT`. Pure-integer parallel path gated on `TINYMIND_ENABLE_QUANTIZATION` (independent of `FLOAT`): `affineToQValueInt` / `qValueToAffineInt` reuse the gemmlowp Q0.31 `multiplyByQuantizedMultiplier` primitive so the deployable freestanding shape `FLOAT=0 STD=0 QUANT=1` can mix Q-format and int8 affine tiers without `<cmath>`
- **`cpp/include/tinymind_fp16.hpp`** -- software-only `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) storage structs over `uint16_t`. Conversion helpers handle normals, subnormals, Inf, NaN. SIMD specialization via Phase 14's `simd_neon_fp16.hpp` on NEON FEAT_FP16 targets
- **Bridge variants** -- `fp16ToAffineI8`, `affineI8ToFp16`, `bf16ToAffineI8`, `affineI8ToBf16` when `TINYMIND_ENABLE_FP16=1`

### SIMD Performance Backend (optional, `TINYMIND_ENABLE_SIMD_*=1`)

ISA-capability-gated SIMD specializations on the inner reduction loop of `QDense`, `QConv2D`, `QConv2DPerChannel`. Every gate defaults to `0`; with all gates off the layer bodies fall back to a scalar dispatch that emits **byte-identical** output to the pre-SIMD build.

- **Gates** -- `TINYMIND_ENABLE_SIMD_NEON`, `_NEON_DOTPROD`, `_NEON_FP16`, `_SVE`, `_SVE2`, `_HELIUM_MVE_I`, `_HELIUM_MVE_F`, `_AVX2`, `_AVX_VNNI`, `_AVX512F`, `_AVX512_VNNI`, plus the orthogonal `TINYMIND_ENABLE_OPENMP`
- **Prerequisite chain** -- each `simd_*.hpp` opens with a `static_assert` enforcing Arm's dependency table: `DOTPROD` requires `NEON`; `SVE`/`SVE2`/FP16-vector require `NEON`; `AVX_VNNI` requires `AVX2`; `AVX512_VNNI` requires `AVX512F`; the two Helium gates are M-profile only and mutually exclusive with `NEON`/`SVE`. Misconfiguration fails at compile time
- **Dispatch** -- public entry point `tinymind::simd::int8DotWithZeroPoint` in `cpp/include/simd/simd_dispatch.hpp`. Backend precedence on x86: `AVX512_VNNI > AVX512F > AVX_VNNI > AVX2 > scalar`; on Arm: `NEON_DOTPROD > NEON > SVE > HELIUM_MVE_I > scalar`. `activeBackendName()` reports the resolved choice
- **Bit-exactness** -- every integer backend is bit-exact with the scalar reference. AVX2 avoids `PMADDUBSW` (saturates); AVX-VNNI / AVX-512-VNNI use the uint8-shift trick so `VPDPBUSD` reduces exactly. Float SIMD reductions are not bit-exact -- invariant applies to integer paths only
- **Threading** -- `cpp/include/threading.hpp` exposes `TINYMIND_PARALLEL_FOR_OUTER` (expands to `#pragma omp parallel for` when `TINYMIND_ENABLE_OPENMP=1`, nothing otherwise). Wired on the output-filter loop of `QConv2D` / `QConv2DPerChannel`. Orthogonal to every SIMD gate; caller passes `-fopenmp`
- **No runtime CPU dispatch** -- library compiles for one ISA per build; fat-binary dispatch is the caller's problem. No `__builtin_cpu_supports` / `getauxval` in library headers; the build system maps `-march=` flags to `TINYMIND_ENABLE_SIMD_*=1`

### Importer Tooling

- **`apps/import_pytorch/tinymind_import.py`** -- PyTorch -> TinyMind int8 importer (pure-numpy core). Caller assembles `Dense` / `Conv2D` / `BatchNorm2D` / `ReLU` / `Sigmoid` / `Tanh` / `Softmax` descriptors carrying numpy weights from `torch.state_dict`. `fuse_layers` folds Conv2D-then-BatchNorm pairs. `calibrate` streams data through the float reference with `MinMaxObserver` / `PercentileObserver` / `KLDivergenceObserver` per layer. `emit_weights_header` writes a TinyMind-format `weights.hpp`. No PyTorch dependency at module import
- **`apps/import_onnx/tinymind_import_onnx.py`** -- QDQ-format ONNX importer. Walks `QuantizeLinear` / `DequantizeLinear` / `QLinearConv` / `QLinearMatMul` / `Relu` / `Sigmoid` / `Tanh` / `Softmax` from a model produced by `onnxruntime.quantization.quantize_static`, emits the same TinyMind-format `weights.hpp`. The `onnx` package is imported lazily inside `parse_onnx_model`

### Activation Functions

| Function | Policy Class | Range |
|----------|-------------|-------|
| Linear | `LinearActivationPolicy` | (-inf, inf) |
| ReLU | `ReluActivationPolicy` | [0, inf) |
| Capped ReLU | `CappedReluActivationPolicy` | [0, max] |
| Sigmoid | `SigmoidActivationPolicy` | (0, 1) |
| Tanh | `TanhActivationPolicy` | (-1, 1) |
| ELU | `EluActivationPolicy` | (-1, inf) |
| GELU | `GeluActivationPolicy` | (-0.17, inf) |
| SiLU | `SiLUActivationPolicy` | (-0.28, inf) |
| Softmax | `SoftmaxActivationPolicy` | (0, 1) per class |

Fixed-point activations use pre-computed lookup tables for speed. Floating-point activations use standard math functions.

### Q-Learning

- Tabular Q-learning with configurable reward and learning policies
- Deep Q-Network (DQN) with neural network function approximation
- Experience replay buffer for DQN training
- Dyna-Q hybrid model-free/model-based learning

### Benchmark Harness (`cpp/include/bench/`)

- **`bench::readCycleCounter()`** -- reads ARM Cortex-M `DWT->CYCCNT` when built with `-DTINYMIND_BENCH_CORTEX_M`, falls back to `std::chrono::steady_clock` nanoseconds on the host
- **`bench::paintStack` / `bench::stackHighWater`** -- canary-based stack watermarking for worst-case RAM measurement on MCUs
- **`bench::LayerStat` + `writeHeader/writeRow`** -- CSV layer stats (name, weight bytes, activation bytes, cycles) that target any sink with `operator<<` (works with `std::ostream` on host and a minimal UART wrapper on MCU, no `<iostream>` dependency required)
- See [`examples/kws_cortex_m/`](examples/kws_cortex_m/) for an end-to-end KWS-style pipeline using the harness, and [`examples/kws_cortex_m_int8/`](examples/kws_cortex_m_int8/) for the int8 quantized counterpart with comparable CSV output

## Quick Start

### Feed-Forward Network (XOR)

```cpp
#include "neuralnet.hpp"
#include "activationFunctions.hpp"

// Define a Q8.8 fixed-point XOR network
typedef tinymind::QValue<8, 8, true> ValueType;
typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>> TransferFunctions;

// 2 inputs, 3 hidden neurons, 1 output
typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<3>, 1,
    TransferFunctions> XorNetwork;

XorNetwork nn;
ValueType inputs[2], target[1];

// Training loop
for (int epoch = 0; epoch < 10000; ++epoch) {
    inputs[0] = 1; inputs[1] = 0; target[0] = 1;
    nn.feedForward(inputs);
    nn.trainNetwork(target);
}

// Inference
nn.feedForward(inputs);
ValueType output[1];
nn.getLearnedValues(output);
```

### Kolmogorov-Arnold Network (XOR)

```cpp
#include "kan.hpp"

// Define a Q8.8 fixed-point KAN
typedef tinymind::QValue<8, 8, true> ValueType;
typedef tinymind::KanTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    1> TransferFunctions;  // 1 output neuron

// 2 inputs, 5 hidden neurons, 1 output
// GridSize=5, SplineDegree=1 (piecewise linear, best for fixed-point)
typedef tinymind::KolmogorovArnoldNetwork<
    ValueType, 2, 1, 5, 1,
    TransferFunctions,
    true,  // trainable
    1,     // batch size
    5,     // grid size
    1      // spline degree
> KanNetwork;

KanNetwork kan;
ValueType inputs[2], target[1], output[1];

// Training loop (same API as MultilayerPerceptron)
for (int epoch = 0; epoch < 10000; ++epoch) {
    inputs[0] = 1; inputs[1] = 0; target[0] = 1;
    kan.feedForward(inputs);
    kan.trainNetwork(target);
}

// Inference
kan.feedForward(inputs);
kan.getLearnedValues(output);
```

### LSTM Network (Sequence Prediction)

```cpp
#include "neuralnet.hpp"

// Floating-point LSTM with 16 hidden neurons
typedef tinymind::LstmNeuralNetwork<double, 1,
    tinymind::HiddenLayers<16>, 1,
    FloatTransferFunctions> LstmNetwork;

LstmNetwork nn;
double input[1], target[1], output[1];

// Sequential training: feed one timestep at a time
for (int epoch = 0; epoch < 100000; ++epoch) {
    nn.resetState();  // clean state each epoch
    for (int t = 0; t < sequenceLength - 1; ++t) {
        input[0] = sequence[t];
        target[0] = sequence[t + 1];
        nn.feedForward(input);
        nn.trainNetwork(target);
    }
}
```

### Multi-Layer LSTM

```cpp
// Two LSTM hidden layers: 16 neurons -> 8 neurons -> output
typedef tinymind::LstmNeuralNetwork<double, 2,
    tinymind::HiddenLayers<16, 8>, 1,
    FloatTransferFunctions> DeepLstmNetwork;

// Three LSTM hidden layers
typedef tinymind::LstmNeuralNetwork<double, 2,
    tinymind::HiddenLayers<32, 16, 8>, 1,
    FloatTransferFunctions> DeeperLstmNetwork;
```

### GRU Network

```cpp
#include "neuralnet.hpp"

// GRU with 8 hidden neurons -- ~25% smaller than equivalent LSTM
typedef tinymind::GruNeuralNetwork<double, 2,
    tinymind::HiddenLayers<8>, 1,
    FloatTransferFunctions> GruNetwork;

GruNetwork nn;
double input[2], target[1], output[1];

// Sequential training (same API as LSTM)
for (int epoch = 0; epoch < 10000; ++epoch) {
    nn.resetState();
    for (int t = 0; t < sequenceLength - 1; ++t) {
        input[0] = sequence[t];
        input[1] = sequence[t + 1];
        target[0] = sequence[t + 2];
        nn.feedForward(input);
        nn.trainNetwork(target);
    }
}
```

### Liquid Cell (CfC, Irregular Sampling)

```cpp
#include "cfc.hpp"
using tinymind::cfc::CfCCell;

// 1 input, 4 state neurons, backbone width 8. Caller owns the flat parameter
// array (type double here) and the state buffer; step<S> is scalar-templated.
typedef CfCCell<1, 4, 8> Cell;
double params[Cell::NumParams];   // [W_bx][W_bh][b_b][W1 b1][W2 b2][WA bA][WB bB]
// ... load trained params ...

double state[4] = {0, 0, 0, 0};
for (size_t t = 0; t < sequenceLength; ++t) {
    double in = sample[t];
    double next[4];
    double ts = elapsed[t];            // per-step elapsed time feeds the time-gate
    Cell::step<double>(params, &in, state, next, ts);
    for (size_t i = 0; i < 4; ++i) state[i] = next[i];
    // `state` is the cell output; add a readout layer downstream
}
// Train through the existing autodiff: build the loss in RevVar and call
// pinn::sgdStepReverse (define TINYMIND_CFC_REVERSE_TRAINING=1). The LTC cell
// (cpp/ltc.hpp) and int8 QCfCCell (cpp/qcfc.hpp) follow the same pattern.
```

### Training with Gradient Clipping and Weight Decay

```cpp
#include "fixedPointTransferFunctions.hpp"

typedef tinymind::QValue<8, 8, true> ValueType;

// Enable gradient clipping [-1, 1] and L2 weight decay (lambda ~ 1/256)
typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    1,                                                  // NumberOfOutputNeurons
    tinymind::DefaultNetworkInitializer<ValueType>,     // initializer
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>, // error calculator
    tinymind::ZeroToleranceCalculator<ValueType>,       // zero tolerance
    tinymind::GradientClipByValue<ValueType>,           // gradient clipping
    tinymind::L2WeightDecay<ValueType>,                 // weight decay
    tinymind::StepDecaySchedule<ValueType, 5000>        // LR decay every 5000 steps
> TransferFunctions;

typedef tinymind::NeuralNetwork<ValueType, 2, tinymind::HiddenLayers<5>, 1,
    TransferFunctions> RegularizedNetwork;
```

### 1D Convolution (Sensor Feature Extraction)

```cpp
#include "conv1d.hpp"

// 100-point sensor input, kernel=5, stride=2, 8 filters
typedef tinymind::Conv1D<double, 100, 5, 2, 8> ConvType;
// Output: 8 filters * 48 positions = 384 features

ConvType conv;
conv.initializeWeights<RandomNumberGenerator>();

double sensorData[100];
double features[ConvType::OutputSize];  // 384

conv.forward(sensorData, features);

// Feed features into a classifier
typedef tinymind::NeuralNetwork<double, 384, tinymind::HiddenLayers<32>, 4,
    TransferFunctions> Classifier;
Classifier nn;
nn.feedForward(features);
```

### Conv1D + Pool1D + Dropout Pipeline

```cpp
#include "conv1d.hpp"
#include "pool1d.hpp"
#include "dropout.hpp"

// Conv1D: 100-point input, kernel=5, stride=1, 4 filters -> 96 * 4 = 384 outputs
typedef tinymind::Conv1D<double, 100, 5, 1, 4> ConvType;

// MaxPool1D: pool size 2, stride 2, 4 channels -> 48 * 4 = 192 outputs
typedef tinymind::MaxPool1D<double, ConvType::OutputLength, 2, 2, 4> PoolType;

// Dropout: 50% dropout on the 192 pooled features
typedef tinymind::Dropout<double, PoolType::OutputSize, 50> DropoutType;

ConvType conv;
PoolType pool;
DropoutType dropout;

double sensorData[100];
double convOut[ConvType::OutputSize];   // 384
double poolOut[PoolType::OutputSize];   // 192
double dropOut[PoolType::OutputSize];   // 192

// Forward pipeline
conv.forward(sensorData, convOut);
pool.forward(convOut, poolOut);
dropout.forward(poolOut, dropOut);  // applies mask during training

// Switch to inference (no dropout)
dropout.setTraining(false);
dropout.forward(poolOut, dropOut);  // identity pass-through
```

### Depthwise-Separable 2D CNN (Keyword Spotting)

```cpp
#include "conv2d.hpp"
#include "depthwiseconv2d.hpp"
#include "pointwiseconv2d.hpp"
#include "pool2d.hpp"

// Input: 20x20x1 (e.g., MFCC tile). Output: 10 class logits.
using Conv1 = tinymind::Conv2D<float, 20, 20, 1, 3, 3, 1, 1, 8>;   // -> 18x18x8
using Pool1 = tinymind::MaxPool2D<float, 18, 18, 8, 2, 2>;          // -> 9x9x8
using Dw    = tinymind::DepthwiseConv2D<float, 9, 9, 8, 3, 3>;      // -> 7x7x8
using Pw    = tinymind::PointwiseConv2D<float, 7, 7, 8, 16>;        // -> 7x7x16
using Gap   = tinymind::GlobalAvgPool2D<float, 7, 7, 16>;           // -> 16
using Dense = tinymind::PointwiseConv2D<float, 1, 1, 16, 10>;       // -> 10

Conv1 conv1; Pool1 pool1; Dw dw; Pw pw; Gap gap; Dense dense;

float input[20 * 20];
float b1[Conv1::OutputSize], b2[Pool1::OutputSize], b3[Dw::OutputSize];
float b4[Pw::OutputSize],    b5[Gap::OutputSize],   logits[Dense::OutputSize];

conv1.forward(input, b1);
pool1.forward(b1, b2);
dw.forward(b2, b3);
pw.forward(b3, b4);
gap.forward(b4, b5);
dense.forward(b5, logits);
```

See [`examples/kws_cortex_m/`](examples/kws_cortex_m/) for the full runnable version with per-layer cycle counts and a port stub for Cortex-M targets.

### Linear Self-Attention (Sequence Processing)

```cpp
#include "selfattention1d.hpp"

// 32 time steps, 16-dim embedding, 8-dim projections
typedef tinymind::SelfAttention1D<double, 32, 16, 8> AttnType;

AttnType attn;

// Set projection weights (W_q, W_k, W_v)
for (size_t proj = 0; proj < 3; ++proj)
    for (size_t r = 0; r < 16; ++r)
        for (size_t c = 0; c < 8; ++c)
            attn.setProjectionWeight(proj, r, c, randomWeight());

double sequence[32 * 16];  // input: 32 time steps x 16 features
double attended[32 * 8];   // output: 32 time steps x 8 features

attn.forward(sequence, attended);

// Feed into a classifier
classifier.feedForward(attended);
```

### Conv1D + Self-Attention Pipeline

```cpp
#include "conv1d.hpp"
#include "selfattention1d.hpp"

// Conv1D extracts features, self-attention models dependencies
typedef tinymind::Conv1D<double, 128, 5, 2, 4> ConvType;
// Conv output: 4 filters * 62 positions = 248 values
// Reshape as: 62 time steps x 4 features
typedef tinymind::SelfAttention1D<double, 62, 4, 4> AttnType;

ConvType conv;
AttnType attn;

double sensorData[128];
double convOut[ConvType::OutputSize];   // 248
double attnInput[62 * 4];              // reshaped
double attnOut[AttnType::OutputSize];  // 248

conv.forward(sensorData, convOut);
// Reshape filter-major to time-step-major...
attn.forward(attnInput, attnOut);
```

### Self-Attention with Fixed-Point (Q16.16)

```cpp
#include "selfattention1d.hpp"

typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
typedef tinymind::SelfAttention1D<ValueType, 16, 8, 4> AttnType;

AttnType attn;

// Set identity projections
for (size_t proj = 0; proj < 3; ++proj)
{
    for (size_t r = 0; r < 8; ++r)
        for (size_t c = 0; c < 4; ++c)
            attn.setProjectionWeight(proj, r, c,
                (r == c) ? ValueType(1, 0) : ValueType(0));
}

ValueType input[16 * 8];
ValueType output[16 * 4];
attn.forward(input, output);
```

### FFT (Frequency-Domain Feature Extraction)

```cpp
#include "fft1d.hpp"
#include <cmath>

// 128-point FFT for sensor signal analysis
const size_t N = 128;
tinymind::FFT1D<double, N> fft;

// Compute twiddle factors
double cosTable[N / 2], sinTable[N / 2];
for (size_t k = 0; k < N / 2; ++k)
{
    cosTable[k] = std::cos(-2.0 * M_PI * k / N);
    sinTable[k] = std::sin(-2.0 * M_PI * k / N);
}
fft.setTwiddleFactors(cosTable, sinTable);

double real[N] = { /* sensor samples */ };
double imag[N] = {}; // zero for real-valued input

fft.forward(real, imag);

// Compute power spectrum (avoids sqrt for fixed-point compatibility)
double magSq[N];
tinymind::FFT1D<double, N>::magnitudeSquared(real, imag, magSq);

// Feed N/2 frequency bins into a classifier
classifier.feedForward(magSq);
```

### Binary Dense Layer (Multiplication-Free)

```cpp
#include "binarylayer.hpp"

// 64 inputs, 16 outputs -- weights stored as packed bits (128 bytes vs 8 KB full-precision)
tinymind::BinaryDense<double, 64, 16> layer;

// Initialize latent weights (real-valued, used during training)
layer.setLatentWeight(0, 0, 0.5);   // will binarize to +1
layer.setLatentWeight(0, 1, -0.3);  // will binarize to -1
// ... set all latent weights

layer.binarizeWeights();  // pack sign(latent_weight) into bits

double input[64], output[16];
layer.forward(input, output);  // XNOR + popcount, no multiplication

// Training with straight-through estimator (STE)
double outputDeltas[16], inputDeltas[64];
layer.backward(outputDeltas, input, inputDeltas);
layer.updateWeights(-0.01);  // updates latent weights, then re-binarizes
```

### Ternary Dense Layer (Multiply-Free with Sparsity)

```cpp
#include "ternarylayer.hpp"

// 64 inputs, 16 outputs, 50% threshold -- weights are {-1, 0, +1}
tinymind::TernaryDense<double, 64, 16, 50> layer;

// Initialize latent weights
layer.setLatentWeight(0, 0, 0.9);   // large magnitude -> +1 or -1
layer.setLatentWeight(0, 1, 0.01);  // small magnitude -> 0 (pruned)
// ... set all latent weights

layer.ternarizeWeights();  // threshold-based: |w| < thresh*mean -> 0

double input[64], output[16];
layer.forward(input, output);  // conditional add/subtract/skip, no multiply

// Training with STE
double outputDeltas[16], inputDeltas[64];
layer.backward(outputDeltas, input, inputDeltas);
layer.updateWeights(-0.01);
```

### RMSprop Optimizer

```cpp
#include "rmsprop.hpp"

// Floating-point RMSprop
typedef FloatingPointTransferFunctions<
    double, RandomNumberGenerator,
    tinymind::TanhActivationPolicy,
    tinymind::TanhActivationPolicy> BaseTF;

struct RmsPropTF : public BaseTF
{
    typedef tinymind::RmsPropOptimizerFloat<double> OptimizerPolicyType;
};

typedef tinymind::MultilayerPerceptron<double, 2, 1, 5, 1, RmsPropTF> Network;

// Fixed-point RMSprop (Q8.8: decay ≈ 230/256 ≈ 0.898, epsilon ≈ 1/256)
typedef tinymind::QValue<8, 8, true> QType;
typedef tinymind::FixedPointTransferFunctions<
    QType, RandomNumberGenerator<QType>,
    tinymind::TanhActivationPolicy<QType>,
    tinymind::TanhActivationPolicy<QType>,
    1,                                                  // NumberOfOutputNeurons
    tinymind::DefaultNetworkInitializer<QType>,
    tinymind::MeanSquaredErrorCalculator<QType, 1>,
    tinymind::ZeroToleranceCalculator<QType>,
    tinymind::NullGradientClippingPolicy<QType>,
    tinymind::NullWeightDecayPolicy<QType>,
    tinymind::FixedLearningRatePolicy<QType>,
    tinymind::RmsPropOptimizer<QType>                   // RMSprop optimizer
> FixedPointTF;
```

### Truncated BPTT (Recurrent Training)

```cpp
#include "truncatedBPTT.hpp"

// LSTM with truncated BPTT over 5-step windows
typedef tinymind::LstmNeuralNetwork<double, 1,
    tinymind::HiddenLayers<16>, 1,
    FloatTransferFunctions> LstmNetwork;

LstmNetwork nn;
tinymind::TruncatedBPTT<LstmNetwork, 5> trainer;

for (int epoch = 0; epoch < 10000; ++epoch) {
    nn.resetState();
    trainer.reset();
    for (int t = 0; t < sequenceLength - 1; ++t) {
        input[0] = sequence[t];
        target[0] = sequence[t + 1];
        trainer.step(nn, input, target);  // trains every 5 steps
    }
    trainer.flush(nn);  // train on remaining steps
}
```

### Q-Learning (Maze)

```cpp
#include "qlearn.hpp"

typedef tinymind::QLearningEnvironment<
    uint8_t,    // state type
    uint8_t,    // action type
    double,     // value type
    6,          // number of states
    6,          // number of actions
    RandomPolicy,
    LearningPolicy> MazeEnvironment;

MazeEnvironment env;
// Run episodes, update Q-values...
```

### PINN Residual via Forward-Mode Autodiff

```cpp
#include "dual.hpp"
#include "dualActivations.hpp"

using tinymind::Dual;
typedef Dual<double> D1;        // first order:  value + d/dx
typedef Dual<D1>     D2;        // second order: nest once more

// A field u(x, t) -- here a trained PINN's forward pass would go.
template<typename S> S u(const S& x, const S& t) {
    return tinymind::tanh(S(1.5) * x + S(0.3) * t);  // toy single neuron
}

const double x0 = 0.7, t0 = 0.2, nu = 0.3;

// du/dt: seed t, hold x constant.
double u_t = u(D1(x0), D1(t0, 1.0)).deriv;

// d^2u/dx^2: nested dual seeded on x at both levels.
D2 x(D1(x0, 1.0), D1(1.0, 0.0));
D2 t(D1(t0), D1(0.0));
double u_xx = u(x, t).deriv.deriv;

double residual = u_t - nu * u_xx;   // heat-equation PDE residual

// Inference (no autodiff): just evaluate u in double or fixed point.
double u_value = u<double>(x0, t0);
```

## Network Types

| Type | Class | Description |
|------|-------|-------------|
| Feed-forward | `NeuralNetwork` | Standard MLP with configurable layers (`MultilayerPerceptron` alias for uniform layers) |
| 1D Convolution | `Conv1D` | Time-series feature extraction with configurable kernel/stride/filters |
| 2D Convolution | `Conv2D` | NHWC 2D convolution for spectrograms, images, time-frequency tiles |
| Depthwise Conv2D | `DepthwiseConv2D` | Per-channel 2D kernel, no cross-channel mixing (MobileNet block) |
| Pointwise Conv2D | `PointwiseConv2D` | 1x1 Conv2D for channel mixing; doubles as a 1x1-input dense layer |
| Max Pooling | `MaxPool1D` | Downsampling via maximum value selection with argmax tracking |
| Average Pooling | `AvgPool1D` | Downsampling via mean with uniform gradient distribution |
| 2D Max Pool | `MaxPool2D` | 2D downsampling via maximum with argmax tracking |
| 2D Avg Pool | `AvgPool2D` | 2D downsampling via mean with uniform gradient distribution |
| Global Avg Pool | `GlobalAvgPool2D` | Collapse HxW to per-channel mean; replaces flatten-to-dense |
| Dropout | `Dropout` | Inverted dropout regularization with training/inference mode |
| Self-Attention | `SelfAttention1D` | Linear attention with ReLU kernel feature map |
| FFT | `FFT1D` | Radix-2 FFT with scaled butterfly for frequency-domain feature extraction |
| Binary Dense | `BinaryDense` | XNOR+popcount dense layer with 1-bit packed weights |
| Ternary Dense | `TernaryDense` | Multiply-free dense layer with 2-bit packed {-1,0,+1} weights |
| KAN | `KolmogorovArnoldNetwork` | Learnable B-spline activations on edges |
| Elman RNN | `ElmanNeuralNetwork` | Simple recurrent with depth-1 feedback |
| Recurrent | `RecurrentNeuralNetwork` | Configurable recurrent connection depth |
| LSTM | `LstmNeuralNetwork` | Long Short-Term Memory with 4 gates |
| GRU | `GruNeuralNetwork` | Gated Recurrent Unit with 3 gates |

All network types support both fixed-point and floating-point value types.

## Architecture

### Policy-Based Design

Every aspect of the network is controlled by template parameters:

```
NeuralNetwork<
    ValueType,              // QValue<8,8,true>, double, float
    NumberOfInputs,         // compile-time input count
    HiddenLayersDescriptor, // HiddenLayers<N0, N1, ...>
    NumberOfOutputs,        // compile-time output count
    TransferFunctionsPolicy,// activation + training policy
    IsTrainable,            // true/false (inference-only mode)
    BatchSize,              // gradient accumulation batch size
    HasRecurrentLayer,      // enables recurrent connections
    HiddenLayerConfig,      // NonRecurrent/Recurrent/GRU/LSTM
    RecurrentConnectionDepth,
    OutputLayerConfiguration // FeedForward/Classifier(softmax)
>
```

### Zero Overhead

The heterogeneous layer chain (`LayerChain`/`EmptyLayerChain`) compiles to the exact same binary size as uniform array-based storage:

**MLP sizes (double):**

| Configuration | Size (bytes) |
|---|---|
| 2 -> 5 -> 1 (1 hidden) | 1,008 |
| 2 -> 3 -> 1 (Elman RNN) | 1,056 |
| 2 -> 5 -> 1 (non-trainable) | 360 |

**KAN vs MLP XOR comparison (Q8.8):**

| | MLP [2]->[3]->[1] | KAN [2]->[5]->[1] G=5 k=1 |
|---|---|---|
| Trainable | 328 bytes | 1,192 bytes |
| Inference-only | 144 bytes | 416 bytes |
| Trainable params | 13 weights | 120 (coefficients + edge weights) |
| Params per edge | 1 scalar | 8 (6 spline coefficients + w_b + w_s) |

**Liquid cells (LTC / CfC) -- parameter bytes:**

These are pointer-shaped structs over a caller-owned flat parameter array, so the footprint is the parameter array (`NumParams x sizeof(ValueType)`), not object `sizeof` -- not directly comparable to the LSTM/GRU object sizes. Each row includes a `NumState -> 1` linear readout.

| Cell (config) | Total params | Q8.8 | `double` |
|---|---|---|---|
| LTC `LtcCell<2,3>` + readout | 28 | 56 bytes | 224 bytes |
| CfC `CfCCell<2,3,4>` + readout | 88 | 176 bytes | 704 bytes |
| LTC `LtcCell<4,8>` + readout | 129 | 258 bytes | 1,032 bytes |
| CfC `CfCCell<4,8,16>` + readout | 761 | 1,522 bytes | 6,088 bytes |

The int8 `QCfCCell<...,2,3,4>` deployable form is 120 bytes of parameters (68 `int8` weights + 13 `int32` biases) plus the two 256-entry sigmoid/tanh LUTs (512 bytes) shared across the whole model. See [docs/size-comparison.md](docs/size-comparison.md#liquid-cells-continuous-time-ltc--cfc) for the full breakdown and the measurement-basis note.

## Visualizing the examples

Every runnable example writes a **header-row CSV** to its `output/` directory and ships a small `plot.py` that renders an appealing graph of the network's behavior (learning curves, fit overlays, int8-vs-float parity, decision surfaces, per-layer cost/footprint bars, RL trajectories, confusion matrices, PDE fields). The C++ side owns the numbers; the Python side only visualizes — so you can drop the CSV into pandas / a spreadsheet / your own tooling and ignore the scripts.

```bash
cd examples/ltc_sequence && make run && make plot   # writes output/*.csv and a PNG
```

The plot scripts share one style module, [`examples/plotting/tinymind_plot.py`](examples/plotting/tinymind_plot.py) (matplotlib only; headless-safe — it falls back to the Agg backend and just writes the PNG when there is no display). `unit_test/nn/nn_plot.py` is the generic per-column network-trajectory viewer (`make plot-trajectory` in `examples/xor`).

matplotlib is the only extra dependency. If your Python already has it, the plot scripts run as-is; otherwise install it — a venv or pyenv is the clean way to add it without touching the system Python.

## Building

### Requirements

- C++14 or later
- [Boost.Test](https://www.boost.org/doc/libs/release/libs/test/) (for unit tests only)
- Set `BOOST_HOME` environment variable to your Boost installation path

### Build and Run Tests

```bash
# Build and run everything
make check

# Individual test suites
cd unit_test/nn && make clean && make && make run
cd unit_test/qformat && make clean && make && make run
cd unit_test/qlearn && make clean && make && make run
cd unit_test/dual && make clean && make && make run     # forward-mode autodiff
```

### Build Examples

```bash
cd examples/xor && make clean && make
cd examples/kan_xor && make clean && make
cd examples/gru_xor && make clean && make
cd examples/lstm_sinusoid && make clean && make
cd examples/maze && make clean && make
cd examples/dqn_maze && make clean && make
cd examples/kws_cortex_m && make clean && make
cd examples/kws_cortex_m_int8 && make clean && make
cd examples/pytorch_quant/xor && make clean && make
cd examples/predictive_maintenance && make clean && make
cd examples/resnet_block_int8 && make clean && make
cd examples/resnet18_block_int8 && make clean && make
cd examples/mobilenetv2_int8 && make clean && make
cd examples/transformer_encoder_int8 && make clean && make
cd examples/mixed_precision_kws && make clean && make
cd examples/mixed_precision_mlp_int8_qformat && make clean && make
cd examples/import_demo && make clean && make
cd examples/perf_matrix && make clean && make
cd examples/pinn_heat1d && make clean && make          # PINN residual via autodiff
```

### Compiler Flags

- Debug: `-Wall -Wextra -Werror -Wpedantic -ggdb`
- Release: `-Wall -Wextra -Werror -Wpedantic -O3`

### Platform Feature Gates

TinyMind compiles cleanly on freestanding embedded targets that lack an FPU, a hosted C++ stdlib, or a C runtime `rand()`. Preprocessor macros control which dependencies are pulled in. **All default to 0** so embedded targets get the strictest configuration out of the box; hosted users opt in via `-DTINYMIND_ENABLE_*=1`. The gates are orthogonal — pick exactly the subset your toolchain (and ISA) provides.

| Macro | What it enables |
|---|---|
| `TINYMIND_ENABLE_FLOAT` | `float`/`double` as `ValueType` (in addition to `QValue`); `ValueParser`/`ValueConverter` float specializations |
| `TINYMIND_ENABLE_STD` | `<cmath>`, `<type_traits>`, `namespace std::`; required for the float-typed `Adam`/`RMSprop`/`Xavier` paths (which need `std::sqrt`) |
| `TINYMIND_ENABLE_HOSTED_IO` | `<fstream>`, `<vector>`, `<cstdlib>`, `<cstdio>`; required for `NetworkPropertiesFileManager` weight serialization |
| `TINYMIND_ENABLE_OSTREAMS` | `<ostream>` and `QValue::operator<<` for debug printing |
| `TINYMIND_ENABLE_HOSTED_RAND` | `<cstdlib>` `rand()`/`RAND_MAX`; required by `Dropout` (training mode), `ScheduledSampling`, and `Xavier` |
| `TINYMIND_ENABLE_QUANTIZATION` | Parallel int8 quantization layer family (`cpp/q*.hpp`) plus `Requantizer`. Calibration helpers in `cpp/include/qcalibration.hpp` are additionally gated on `FLOAT && STD` (host-only) |
| `TINYMIND_ENABLE_FP16` | `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) storage tier in `cpp/include/tinymind_fp16.hpp`. Conversion helpers additionally require `TINYMIND_ENABLE_FLOAT` |
| `TINYMIND_ENABLE_INT16_ACCUM` | Wide cell-state for `QLSTMCell` (cell stored as `int16_t` rather than `int8_t`) for long unroll horizons |
| `TINYMIND_ENABLE_SIMD_NEON` | Arm NEON int8 dot-product specialization |
| `TINYMIND_ENABLE_SIMD_NEON_DOTPROD` | Arm NEON FEAT_DOTPROD (`SDOT`/`UDOT`); requires `_NEON` |
| `TINYMIND_ENABLE_SIMD_NEON_FP16` | Arm NEON FEAT_FP16 vector arithmetic; requires `_NEON` |
| `TINYMIND_ENABLE_SIMD_SVE` / `_SVE2` | Arm Scalable Vector Extension; requires `_NEON` |
| `TINYMIND_ENABLE_SIMD_HELIUM_MVE_I` / `_MVE_F` | M-profile Helium MVE (integer/float). Mutually exclusive with `_NEON`/`_SVE` |
| `TINYMIND_ENABLE_SIMD_AVX2` | x86 AVX2 |
| `TINYMIND_ENABLE_SIMD_AVX_VNNI` | AVX-VNNI (`VPDPBUSD`); requires `_AVX2` |
| `TINYMIND_ENABLE_SIMD_AVX512F` | x86 AVX-512 Foundation |
| `TINYMIND_ENABLE_SIMD_AVX512_VNNI` | AVX-512-VNNI; requires `_AVX512F` |
| `TINYMIND_ENABLE_OPENMP` | OpenMP parallelization of the output-filter loop in `QConv2D` / `QConv2DPerChannel`. Orthogonal to every SIMD gate; caller passes `-fopenmp` separately |

With all gates at 0, no header in `cpp/` includes anything beyond `<cstddef>`, `<cstdint>`, and placement `<new>` — all required by freestanding C++. With `FLOAT=1, STD=0` you get an FPU-but-no-stdlib build for float forward-pass inference. With `QUANT=1, FLOAT=0, STD=0` you get the deployable int8 inference shape (no float, no calibration helpers). The `unit_test/embedded` matrix builds and runs **eight corners** (`freestanding`, `no_stdlib`, `no_fpu`, `hosted`, `quant_freestanding`, `fp16_freestanding`, `int16_accum_freestanding`, `simd_disabled`) as part of `make check`. A separate `simd_prereq_regressions` target locks the static_assert prerequisite chain via compile-failure checks (e.g. `AVX_VNNI=1, AVX2=0` must fail to compile).

## Quality Gates

Every push and pull request runs the [Static & Dynamic Analysis](.github/workflows/analysis.yml) workflow plus a separate [CodeQL](.github/workflows/codeql.yml) scan. The bar is split into **blocking** gates (a failure fails the merge) and **advisory** gates (`continue-on-error: true` — findings are surfaced as uploaded report artifacts but never block).

| Gate | Tool | Status | Reproduce locally |
|---|---|---|---|
| Memory / UB sanitizers | Clang ASan + UBSan over all runtime suites + int8 examples | **Blocking** | `make sanitize` |
| Static analysis | cppcheck (warning + portability), all headers | **Blocking** | `make cppcheck` |
| Coverage floor | gcov + lcov, `cpp/` line/function floors | **Blocking** | `make coverage && make coverage-check` |
| Formal proofs | CBMC over the fixed-point `qformat` kernels | **Blocking** | `make -C formal prove` |
| Fuzzing | libFuzzer over int8 kernels, time-boxed, ASan+UBSan | **Blocking** | `make -C fuzz fuzz-ci` |
| Semantic code scan | GitHub CodeQL (C/C++) | **Blocking** | — (cloud) |
| MISRA C:2012 | cppcheck MISRA addon (advisory against C++17 template code) | Advisory | `make misra` |
| Lint + clang static analyzer | clang-tidy (`bugprone-*`, `cert-*`, `clang-analyzer-*`, narrowing/init) | Advisory | `make tidy` |

The clang-tidy gate runs the **Clang Static Analyzer** engine in-process via its `clang-analyzer-*` checks — there is no separate `scan-build` pass. The full enforced matrix (build flags, the eight-corner embedded regression set) runs under `make check`.

## Project Structure

```
tinymind/
  cpp/                          # Core library headers
    neuralnet.hpp               # Neural network templates (~4500 lines)
    kan.hpp                     # Kolmogorov-Arnold Network templates
    bspline.hpp                 # B-spline evaluation engine (De Boor algorithm)
    kanTransferFunctions.hpp    # KAN transfer functions and SiLU activation
    conv1d.hpp                  # 1D convolution layer
    conv2d.hpp                  # 2D convolution layer (NHWC, VALID padding)
    depthwiseconv2d.hpp         # Depthwise 2D convolution (per-channel kernels)
    pointwiseconv2d.hpp         # 1x1 pointwise convolution (channel mixing / dense)
    pool1d.hpp                  # MaxPool1D and AvgPool1D layers
    pool2d.hpp                  # MaxPool2D, AvgPool2D, GlobalAvgPool2D
    selfattention1d.hpp         # Linear self-attention layer
    fft1d.hpp                   # Radix-2 FFT with compile-time bit-reversal tables
    batchnorm.hpp               # Batch normalization (training/inference)
    binarylayer.hpp             # Binary neural network layer (XNOR+popcount)
    ternarylayer.hpp            # Ternary neural network layer ({-1,0,+1} weights)
    qformat.hpp                 # Fixed-point arithmetic
    qlearn.hpp                  # Q-learning and DQN
    qaffine.hpp                 # Affine quantization primitives + Requantizer (TINYMIND_ENABLE_QUANTIZATION)
    qconv2d.hpp                 # Quantized 2D convolution (per-tensor + per-channel)
    qdepthwiseconv2d.hpp        # Quantized depthwise 2D conv (per-channel weight scale, TFLite mandate)
    qpointwiseconv2d.hpp        # Quantized 1x1 pointwise conv (per-tensor + per-channel)
    qpool2d.hpp                 # QMaxPool2D, QAvgPool2D, QGlobalAvgPool2D
    qdense.hpp                  # Quantized fully-connected layer
    qactivations.hpp            # Quantized ReLU / ReLU6 + fused-clamp helpers + int8 sigmoid/tanh LUTs
    qadd.hpp                    # QAdd (TFLite ADD semantics)
    qmul.hpp                    # QMul (elementwise int8 multiply)
    qconcat.hpp                 # QConcat2_2D (channel-axis concat)
    qpad.hpp                    # QPad2D (constant pad using zero_point)
    qbatchnorm.hpp              # QBatchNorm1D / QBatchNorm2D (standalone, non-fused)
    qlayernorm.hpp              # QLayerNorm1D (integer mean/var + qInvSqrtQ30 Newton)
    qsoftmax.hpp                # QSoftmax1D (two-pass int8->int8 via int32 exp LUT)
    qlstm.hpp                   # QLSTMCell (int8 or int16 cell state)
    qgru.hpp                    # QGRUCell (3 gates, reset-before-multiply)
    qfft1d.hpp                  # QFFT1D (int16 radix-2 DIT, Q1.15 twiddles)
    qattention1d.hpp            # QAttention1D (int8 linear attention)
    qattention_softmax.hpp      # QAttentionSoftmax1D (standard softmax attention)
    qmha.hpp                    # QMultiHeadLinearAttention1D
    qbridge.hpp                 # Mixed-precision bridges (qformat <-> affine <-> fp16/bf16/float)
    activationFunctions.hpp     # Activation function policies (9 functions)
    fixedPointTransferFunctions.hpp
    adam.hpp                    # Adam optimizer policy
    rmsprop.hpp                 # RMSprop optimizer policy
    dropout.hpp                 # Inverted dropout regularization layer
    gradientClipping.hpp        # Gradient clipping policies
    weightDecay.hpp             # L2 weight decay policies
    learningRateSchedule.hpp    # Learning rate scheduling policies
    earlyStopping.hpp           # Early stopping convergence monitor
    teacherForcing.hpp          # Scheduled sampling for recurrent training
    truncatedBPTT.hpp           # Truncated BPTT training utility
    networkStats.hpp            # Compile-time network statistics
    xavier.hpp                  # Xavier weight initialization
    lookupTables.cpp            # Pre-computed activation tables (~4.6 MB)
    include/                    # Support headers
      tinymind_platform.hpp     # Platform feature gates (FLOAT/STD/HOSTED_IO/OSTREAMS/HOSTED_RAND/QUANTIZATION/FP16/INT16_ACCUM/SIMD_*/OPENMP)
      tinymind_traits.hpp       # Minimal in-house enable_if / is_floating_point for STD=0 builds
      tinymind_fp16.hpp         # fp16_t / bf16_t storage structs + float converters
      threading.hpp             # TINYMIND_PARALLEL_FOR_OUTER OpenMP macro
      nnproperties.hpp          # Weight file manager (MLP, LSTM, GRU, KAN)
      qcalibration.hpp          # Host-only int8 calibration (RangeObserver, PercentileObserver, KLDivergenceObserver, CLE, fold, ...)
      constants.hpp, limits.hpp, random.hpp, ...
      simd/                     # ISA-capability SIMD specializations (Phase 14)
        simd_dispatch.hpp       # Public entry + active backend selection
        simd_neon.hpp, simd_neon_dotprod.hpp, simd_neon_fp16.hpp
        simd_sve.hpp, simd_sve2.hpp
        simd_helium_mve_i.hpp, simd_helium_mve_f.hpp
        simd_avx2.hpp, simd_avx_vnni.hpp
        simd_avx512f.hpp, simd_avx512_vnni.hpp
      bench/                    # Benchmark harness
        platform.hpp            # Cycle counter (Cortex-M DWT / host chrono) + stack watermarks
        report.hpp              # LayerStat CSV rows and ScopedTimer
  examples/
    xor/                        # MLP XOR gate learning
    kan_xor/                    # KAN XOR gate learning
    gru_xor/                    # GRU XOR gate learning
    lstm_sinusoid/              # LSTM sinusoid prediction
    lstm_sinusoid_float/        # LSTM sinusoid: float (double) vs Q16.16 side by side
    elman_temporal_xor/         # Elman vs MLP on temporal XOR (recurrent memory demo)
    maze/                       # Tabular Q-learning maze solver
    dqn_maze/                   # Deep Q-Network maze solver
    kws_cortex_m/               # Depthwise-separable CNN pipeline with bench harness
    kws_cortex_m_int8/          # int8 quantized counterpart of kws_cortex_m (parallel Q* layers)
    predictive_maintenance/     # Binary classifier on AI4I 2020 dataset (Q16.16 MLP)
    pytorch/                    # PyTorch weight import (MLP + GRU export, Q-format pipeline)
    pytorch_quant/xor/          # PyTorch -> int8 affine quantization end-to-end (XOR)
    resnet_block_int8/          # int8 residual block (Phase 10 per-channel + QAdd)
    resnet18_block_int8/        # int8 ResNet-18-shaped stem + basic block
    mobilenetv2_int8/           # int8 MobileNetV2 inverted-residual sequence
    transformer_encoder_int8/   # int8 transformer encoder block (LayerNorm + attention + dense)
    mixed_precision_kws/        # int8 -> fp16 -> int8 KWS (Phase 9 qbridge)
    mixed_precision_mlp_int8_qformat/ # Hybrid int8 affine <-> Q8.8 (Phase 17 integer bridges)
    import_demo/                # End-to-end Phase 15 importer (3-8-4-2 MLP, three observers + CLE)
    perf_matrix/                # SIMD gate bench (CSV per backend, invariant output_checksum)
  unit_test/
    nn/                         # Neural network tests (171 test cases)
    kan/                        # KAN tests (16 test cases)
    qformat/                    # Fixed-point type tests (static_assert)
    qlearn/                     # Q-learning tests
    quantization/               # int8 path: Requantizer, calibration, Q* layer correctness, SIMD bit-exactness
    lookuptable/                # Activation lookup-table generator regressions
    embedded/                   # Eight-corner (FLOAT, STD, QUANT, FP16, INT16_ACCUM, SIMD_*) regression matrix + simd_prereq_regressions
    integration/                # Golden-byte regression suite for the Phase 16/17 exemplars
  apps/
    activation/                 # Lookup table generator tool
    import_pytorch/             # PyTorch -> TinyMind int8 importer (numpy core)
    import_onnx/                # QDQ-format ONNX -> TinyMind int8 importer
```

## Documentation

- [CLAUDE.md](CLAUDE.md) -- Architecture overview and build commands
- [KAN.md](KAN.md) -- KAN implementation plan and summary
- [LSTM.md](LSTM.md) -- LSTM implementation analysis and improvement roadmap
- [QUANTIZATION.md](QUANTIZATION.md) -- Post-training int8 quantization plan and design notes

## License

MIT License. See individual source files for copyright notices.
