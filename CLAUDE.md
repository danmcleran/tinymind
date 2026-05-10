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

Six preprocessor macros in `cpp/include/tinymind_platform.hpp` control optional hosted dependencies. **All default to 0** (freestanding); hosted Makefiles set `=1` selectively.

- `TINYMIND_ENABLE_FLOAT` — `float`/`double` as `ValueType`
- `TINYMIND_ENABLE_STD` — `<cmath>`, `<type_traits>`, `namespace std::`
- `TINYMIND_ENABLE_HOSTED_IO` — `<fstream>`, `<vector>`, `<cstdlib>`, `<cstdio>` (file weight serialization)
- `TINYMIND_ENABLE_OSTREAMS` — `<ostream>` and `QValue::operator<<`
- `TINYMIND_ENABLE_HOSTED_RAND` — `<cstdlib>` `rand()` (Dropout training mode, ScheduledSampling, Xavier)
- `TINYMIND_ENABLE_QUANTIZATION` — `cpp/q*.hpp` int8 quantization layer family + `Requantizer`. Calibration helpers in `cpp/include/qcalibration.hpp` are additionally gated on `FLOAT && STD` (host-only). Existing fixed-point and float pipelines are unaffected.

`unit_test/embedded` is the regression guard — its Makefile builds five corners of the `(FLOAT, STD, QUANT)` matrix (`freestanding`, `no_stdlib`, `no_fpu`, `hosted`, `quant_freestanding`) and `make check` runs all of them. Float-typed Adam/RMSprop/Xavier require both `FLOAT && STD`. When adding a new header to `cpp/`, include `"include/tinymind_platform.hpp"` first and gate any `<cmath>`/`<type_traits>`/`std::` use; for SFINAE on `is_floating_point`, use `tinymind::enable_if` and `tinymind::is_floating_point` from `cpp/include/tinymind_traits.hpp` so the layer compiles at `STD=0`.

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

- **`qaffine.hpp`** — Affine primitives: `QAffineTensor`, `Requantizer<SrcAccum, DstStorage>`, and the gemmlowp-style `saturatingRoundingDoublingHighMul` / `roundingDivideByPOT` helpers. Pure integer at runtime; freestanding-safe.
- **`qconv2d.hpp`**, **`qdepthwiseconv2d.hpp`** (per-channel weight scale, TFLite mandate), **`qpointwiseconv2d.hpp`**, **`qpool2d.hpp`** (`QMaxPool2D` / `QAvgPool2D` / `QGlobalAvgPool2D`), **`qdense.hpp`** — Quantized layer family. Weights, biases, and Requantizer tables are caller-owned; layer structs are pointer-shaped so the same model can be built once on the host and re-used across many MCU targets.
- **`qactivations.hpp`** — Quantized ReLU / ReLU6, plus `clampForRelu` / `clampForRelu6` helpers that fold the activation into the upstream Requantizer's saturation pass (matches CMSIS-NN runtime efficiency). 256-entry int8 lookup tables for sigmoid and tanh (`buildQSigmoidLUT` / `buildQTanhLUT` build host-side; `qApplyLUT` / `qApplyLUTBuffer` apply pure-integer at runtime).
- **`include/qcalibration.hpp`** — Host-only calibration helpers (`RangeObserver`, `computeAffineParamsAsymmetric` / `Symmetric`, `computePerChannelSymmetricScales`, `quantizeBuffer`, `buildRequantizer`). Gated on `TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD` so the deployable inference binary never pulls in float math.

The deployable target shape is `TINYMIND_ENABLE_QUANTIZATION=1, TINYMIND_ENABLE_FLOAT=0, TINYMIND_ENABLE_STD=0`: int8 weights/activations, int32 accumulators, integer requantization, no `<cmath>`. The `unit_test/embedded` matrix exercises this corner as `quant_freestanding`. The `unit_test/quantization` Boost.Test suite covers the math (Requantizer round-trip, per-channel depthwise, calibration), and `examples/kws_cortex_m_int8/` is a side-by-side counterpart to `examples/kws_cortex_m/` with directly comparable CSV cycle/byte reports.

Non-goals in this phase: no QAT (post-training only), no int4/mixed precision, no conv+bn+relu fusion pass.

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
- **`quantization/`** — Boost.Test unit tests for the int8 quantization path: Requantizer round-trip, per-tensor / per-channel calibration, QConv2D / QDepthwise / QPointwise / QPool / QDense forward passes against a float reference. Builds with `TINYMIND_ENABLE_QUANTIZATION=1`.
- **`embedded/`** — Cross-corner regression matrix. Builds the smoke source under five `(FLOAT, STD, QUANT)` configurations including `quant_freestanding` (`FLOAT=0 STD=0 QUANT=1`) which is the deployable inference shape for an int8 MCU target.

### Examples (`examples/`)

- **`xor/`** — XOR gate learned by a small neural network; includes a Python plotting script
- **`maze/`** and **`dqn_maze/`** — Maze solving via Q-learning and deep Q-networks
- **`pytorch/`** — Exports weights from a PyTorch model and imports them into a TinyMind C++ network for inference (Q-format pipeline)
- **`pytorch_quant/xor/`** — Affine int8 counterpart to `pytorch/xor/`. PyTorch float training + per-tensor calibration + `weights.hpp` emission, then a pure-integer C++ forward pass through `QDense` + `qrelu` + `QDense` + int8 sigmoid LUT
- **`kws_cortex_m/`** — Keyword-spotting-style pipeline built from `Conv2D` → `MaxPool2D` → `DepthwiseConv2D` → `PointwiseConv2D` → `GlobalAvgPool2D` → dense, with a CSV cycles/bytes report from the bench harness. Host runner; includes a `port_stub.hpp` sketch for porting to a Cortex-M target.
- **`kws_cortex_m_int8/`** — int8 quantized counterpart of `kws_cortex_m`. Same pipeline shape, same CSV report format, but every layer is replaced with the `cpp/q*.hpp` family. Demonstrates host-side calibration via `qcalibration.hpp` plus a pure-integer forward path. Use it for direct cycle/byte comparisons against the float pipeline.

### Apps (`apps/activation/`)

Standalone tool that generates the lookup table values used in `lookupTables.cpp`.

### Benchmark harness (`cpp/include/bench/`)

- **`platform.hpp`** — `bench::readCycleCounter()` (DWT CYCCNT on Cortex-M, `std::chrono` host fallback) and `bench::paintStack` / `bench::stackHighWater` for MCU stack watermarking. Enable the MCU path with `-DTINYMIND_BENCH_CORTEX_M`.
- **`report.hpp`** — `bench::LayerStat` CSV row type, `bench::writeHeader` / `bench::writeRow` for any sink that supports `operator<<`, and a `bench::ScopedTimer` for per-layer measurements. See `examples/kws_cortex_m/` for an end-to-end use.
