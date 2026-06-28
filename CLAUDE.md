# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Long-form architecture reference:** the optional quantization / continuous-time /
> SIMD tiers, the full test matrix, every example, and the apps are documented in
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). This file keeps the load-bearing
> rules (build commands, feature gates, core library map, design pattern); reach for
> the reference when you need detail on a specific `cpp/q*.hpp` layer, exemplar, or
> phase.

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

The optional tiers below are summarized only in `docs/ARCHITECTURE.md`: int8 **Quantization** (`cpp/q*.hpp` family + `qcalibration.hpp`), **Recurrent Quantization** (`QLSTMCell` / `QGRUCell`), **Liquid Time-Constant** (`ltc.hpp`) and **Closed-form Continuous-time** (`cfc.hpp` / `qcfc.hpp`) cells, int8 **Attention + FFT** (`qattention*.hpp` / `qfft1d.hpp` / `qmha.hpp`), int8 **Transformer Decoder / seq2seq** (`qcausalattention1d.hpp` / `qcausalattention_softmax.hpp` / `qcrossattention.hpp` / `qkvcache.hpp`), int8 **State-Space / S4-lite** (`qssm.hpp`), **Mixed-Precision Bridges** (`qbridge.hpp` / `tinymind_fp16.hpp`), the **SIMD Performance Backend** (`cpp/include/simd/`), and the calibration/importer tooling (`apps/import_pytorch`, `apps/import_onnx`).

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

## Tests, Examples & Apps

Full detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Quick map:

- **`unit_test/`** — `nn/` (NN correctness), `qformat/` (compile-time `static_assert`), `qlearn/`, `quantization/` (int8 path), `embedded/` (cross-corner regression matrix + `simd_prereq_regressions`), `integration/` (Phase 16 golden-byte exemplar suite), `pinn/`, `ltc/`, `cfc/`.
- **`examples/`** — every runnable example writes a header-row CSV to `output/` and ships a `plot.py` using the shared `examples/plotting/tinymind_plot.py` style module (`make plot`). matplotlib goes in an isolated env (venv/pyenv), never system Python. int8 exemplars expose `make run` / `make bench` / `make golden`.
- **`apps/`** — `activation/` (LUT generator for `lookupTables.cpp`), `import_pytorch/` + `import_onnx/` (Phase 15 int8 importers).

## Benchmark harness (`cpp/include/bench/`)

- **`platform.hpp`** — `bench::readCycleCounter()` (DWT CYCCNT on Cortex-M, `std::chrono` host fallback) and `bench::paintStack` / `bench::stackHighWater` for MCU stack watermarking. Enable the MCU path with `-DTINYMIND_BENCH_CORTEX_M`.
- **`report.hpp`** — `bench::LayerStat` CSV row type, `bench::writeHeader` / `bench::writeRow` for any sink that supports `operator<<`, and a `bench::ScopedTimer` for per-layer measurements. See `examples/kws_cortex_m/` for an end-to-end use.
