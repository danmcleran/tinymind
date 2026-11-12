# cpp

The library. Repo-wide rules (C++14 floor, SGD rounding, style, workflow) are in
the root `AGENTS.md`; the optional tiers are described in `docs/ARCHITECTURE.md`
and the SIMD backends in `include/simd/AGENTS.md`.

## Header rules

- When adding a new header to `cpp/`, include `"include/tinymind_platform.hpp"` first and gate any `<cmath>`/`<type_traits>`/`std::` use; for SFINAE on `is_floating_point`, use `tinymind::enable_if` and `tinymind::is_floating_point` from `cpp/include/tinymind_traits.hpp` so the layer compiles at `STD=0`.
- Float-typed Adam/RMSprop/Xavier require both `FLOAT && STD`.
- Every header uses `#pragma once` (no `#ifndef` guards) and carries its own `<cstddef>`/`<cstdint>` includes; `header-selfcheck` compiles each one standalone.
- No heap allocation on the embedded paths: never add `malloc` or a heap `new`. The `heapfree` gate `nm`-asserts that the freestanding and quantization objects reference neither; `<vector>` and file I/O exist only behind `TINYMIND_ENABLE_HOSTED_IO`.
- All code lives in the `tinymind` namespace; internal helpers are in `tinymind::detail`.
- `cpp/lookupTables.cpp` is the only `.cpp` file in the core library. Fixed-point LUT activations are switched on per format with `-DTINYMIND_USE_<FUNCTION>_<I>_<F>=1` (for example `TINYMIND_USE_TANH_8_8`, `TINYMIND_USE_SIGMOID_16_16`), and the binary must link `lookupTables.cpp`; see the Makefile of any example for the flags.
- An `int8_t` store may alias any object, `*this` included, so a member read inside a loop that stores int8 is reloaded every iteration unless it is copied to a local first (`QDense::forward` explains it). Copy a member out only after measuring the change on the F411: in `QAvgPool2D` the copied clamp bounds took registers from the window pointers, which spilled to the stack, and the layer got 7% slower. It makes no difference when the layer object is a visible `const` with constant initializers, as the importers emit, because the compiler already folds those members -- as long as `forward()` is small enough to be inlined into its caller. Keep a rarely taken fallback out of line with `TINYMIND_NOINLINE` (see `QAdd::forward`): with the reference loop inlined beside the fast loop, `QAdd::forward` stopped being inlined, lost its constants, and ran 41% slower on the F411.
- A kernel that calls `Requantizer::apply` once per output carries its saturating path's code into the loop even when it never runs. On Thumb-1 at `-O2` that spilled the dense layers' dot product (the L031 MLP ran 36% slower after #230), so `QDense` and the single-pixel pointwise layers check `oneRounding()` once per layer and, under `TINYMIND_SPLIT_REQUANTIZER_FORMS` (default on for Thumb-1 only, and on in `unit_test/quantization`), run a copy of the loop that calls `applyOneRounding()`. On the Cortex-M4 the split measured slower over a whole inlined model, so other targets keep the single loop and compile to the same code as before. The loops are written inline, not as templated helpers: a helper instantiated per layer type that some test never drives fails the coverage gate's 100% function floor.
- Do not put `__restrict` on a public signature. Several APIs promise that buffers may overlap (`cfc.hpp` and `ltc.hpp` state, `foldBatchNorm` in place, the FFTs, element-wise layers with `out == a`), and breaking a restrict promise is silent undefined behaviour that no sanitizer reports.
- Warning names that exist in only one compiler must use `TINYMIND_DISABLE_WARNING_GCC_ONLY` / `_CLANG_ONLY` from `include/compiler.h` — the plain `TINYMIND_DISABLE_WARNING` form errors on the compiler that does not own the name (`-Wunknown-warning-option` under `-Werror`).

## Platform Feature Gates

Preprocessor macros in `cpp/include/tinymind_platform.hpp` control optional hosted dependencies and per-ISA SIMD specializations. **All default to 0** (freestanding, scalar); hosted Makefiles set `=1` selectively.

- `TINYMIND_ENABLE_FLOAT` — `float`/`double` as `ValueType`
- `TINYMIND_ENABLE_STD` — `<cmath>`, `<type_traits>`, `namespace std::`
- `TINYMIND_ENABLE_HOSTED_IO` — `<fstream>`, `<vector>`, `<cstdlib>`, `<cstdio>` (file weight serialization)
- `TINYMIND_ENABLE_OSTREAMS` — `<ostream>` and `QValue::operator<<`
- `TINYMIND_ENABLE_HOSTED_RAND` — `<cstdlib>` `rand()` (Dropout training mode, ScheduledSampling, Xavier)
- `TINYMIND_ENABLE_QUANTIZATION` — `cpp/q*.hpp` int8 quantization layer family + `Requantizer`. Calibration helpers in `cpp/include/qcalibration.hpp` are additionally gated on `FLOAT && STD` (host-only). Existing fixed-point and float pipelines are unaffected.
- `TINYMIND_ENABLE_FP16` — Phase 9 half-precision storage tier: `fp16_t` (IEEE 754 binary16) and `bf16_t` (bfloat16) in `cpp/include/tinymind_fp16.hpp`, with float conversion helpers. Storage tier; Phase 14's `simd_neon_fp16.hpp` provides the matching vector specialization. Conversion helpers additionally require `TINYMIND_ENABLE_FLOAT`.
- `TINYMIND_ENABLE_INT16_ACCUM` — Phase 12 wide carry-state for `QLSTMCell` (cell-state stored as `int16_t` rather than `int8_t`). The cell template itself only needs the C++ type, so the gate purely controls whether the embedded regression matrix advertises the int16 corner; targets without an int16 storage path simply do not instantiate the wide template.
- `TINYMIND_ENABLE_SIMD_*` — Phase 14 ISA-capability SIMD gates (NEON, NEON dot product, NEON fp16, SVE, SVE2, Helium MVE-I/MVE-F, Arm DSP, AVX2, AVX-VNNI, AVX-512F, AVX-512 VNNI). Gates name ISA extensions, never CPU models; scalar fallback is byte-identical when every gate is off. Full list, prerequisite chain and validation: `cpp/include/simd/AGENTS.md`.
- `TINYMIND_ENABLE_OPENMP` — Phase 14 OpenMP parallelization of the output-filter loop in `QConv2D` / `QConv2DPerChannel` via `TINYMIND_PARALLEL_FOR_OUTER` in `cpp/include/threading.hpp`. Orthogonal to every SIMD gate; caller passes `-fopenmp` separately.

`unit_test/embedded` is the regression guard for these gates; see `unit_test/embedded/AGENTS.md`.

## Architecture Overview

tinymind is a **header-only C++ template library** for neural networks and Q-learning, designed for embedded systems with no FPU, GPU, or vectorized instruction requirements. The design is inspired by Andrei Alexandrescu's policy-based design from *Modern C++ Design*.

The optional tiers below are summarized only in `docs/ARCHITECTURE.md`: int8 **Quantization** (`cpp/q*.hpp` family + `qcalibration.hpp`), **Recurrent Quantization** (`QLSTMCell` / `QGRUCell`), **Liquid Time-Constant** (`ltc.hpp`) and **Closed-form Continuous-time** (`cfc.hpp` / `qcfc.hpp`) cells, int8 **Attention + FFT** (`qattention*.hpp` / `qfft1d.hpp` / `qmha.hpp`), int8 **Transformer Decoder / seq2seq** (`qcausalattention1d.hpp` / `qcausalattention_softmax.hpp` / `qcrossattention.hpp` / `qkvcache.hpp`), int8 **State-Space / S4-lite** (`qssm.hpp`), int8 **Decision Trees / GBDT** (`qtree.hpp`), int8 **Neuro-Fuzzy** (`qanfis.hpp`), **Mixed-Precision Bridges** (`qbridge.hpp` / `tinymind_fp16.hpp`), the **SIMD Performance Backend** (`cpp/include/simd/`), and the calibration/importer tooling (`apps/import_pytorch`, `apps/import_onnx`, `apps/import_trees`).

### Core Library (`cpp/`)

- **`qformat.hpp`** — Fixed-point arithmetic templates. Supports Q8.8, Q16.16, Q24.8, Q32.32, and other formats (up to 128-bit). Also supports `float` as a value type.
- **`neuralnet.hpp`** — The main neural network template (~5000 lines). Supports feed-forward and recurrent networks. Network topology, value type, activation function, and other policies are all template parameters.
- **`qlearn.hpp`** — Q-learning and DQN reinforcement learning implementation.
- **`activationFunctions.hpp`**, **`fixedPointTransferFunctions.hpp`** — Activation functions (sigmoid, tanh, ReLU, etc.) with separate implementations for fixed-point vs. float.
- **`lookupTables.cpp`** — Large pre-computed lookup tables for fixed-point activation functions (sigmoid, tanh, exp, log, sin, cos). This is the only `.cpp` file in the core library; everything else is headers.
- **`cpp/include/nnproperties.hpp`** — Defines the property/policy classes used to configure neural network templates (layer sizes, learning rates, activation functions, etc.).
- **`cpp/include/tinymind_rounding.hpp`** — `multiplyRoundNearest` / `multiplyRoundTowardZero`, the only way an SGD step is built (see the root `AGENTS.md`).
- **`kan.hpp`**, **`bspline.hpp`**, **`kanTransferFunctions.hpp`** — Kolmogorov-Arnold networks (see `KAN.md`).
- **`adam.hpp`**, **`rmsprop.hpp`**, **`gradientClipping.hpp`**, **`weightDecay.hpp`**, **`learningRateSchedule.hpp`**, **`earlyStopping.hpp`**, **`teacherForcing.hpp`**, **`xavier.hpp`** — optional training policies, detected at compile time by SFINAE in `tinymind::detail`; a network opts in through its template parameters, never a runtime check.
- **`dual.hpp`**, **`multidual.hpp`**, **`revdual.hpp`**, **`pinn.hpp`** — forward- and reverse-mode automatic differentiation and the PINN building blocks.
- **`moe.hpp`** — `MixtureOfExperts` with a pluggable expert type.

### Standalone Composable Layers (`cpp/`)

These layers sit outside the neural network template and can be chained into pipelines:

- **`conv1d.hpp`** — 1D convolution layer for time-series feature extraction.
- **`pool1d.hpp`** — `MaxPool1D` and `AvgPool1D` for downsampling.
- **`upsample1d.hpp`** — `UpsampleNearest1D` (repeat; no arithmetic, holds at `FLOAT=0`/`STD=0`) and `UpsampleLinear1D` (interpolated; QValue/float arithmetic) for integer-factor upsampling. The decoder-side counterpart to `pool1d.hpp`; a cheap, checkerboard-free alternative to a transposed convolution.
- **`conv2d.hpp`** — 2D convolution for spectrograms/images, NHWC layout, VALID padding.
- **`depthwiseconv2d.hpp`** / **`pointwiseconv2d.hpp`** — MobileNet-style depthwise-separable blocks.
- **`pool2d.hpp`** — `MaxPool2D`, `AvgPool2D`, and `GlobalAvgPool2D` (GAP replaces the big flatten-to-dense matrix).
- **`anfis.hpp`** — Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno), inference only. `Anfis<ValueType, NumberOfInputs, NumberOfMembershipFunctionsPerInput, NumberOfRules, MembershipFunctionPolicy, FirstOrderConsequent, NumberOfOutputs>` plus four membership-function policies (`Triangular`, `Trapezoidal`, `GeneralizedBell`, `Gaussian`) and `AnfisFullGridRuleTable`. Rule base is an explicit `uint8_t` antecedent table (supports offline pruning + don't-care antecedents), not an implicit `M^N` grid. Triangular / trapezoidal / bell are arithmetic-only and hold at `FLOAT=0`/`STD=0`; the Gaussian rides the integer exp LUT. Training (hybrid least-squares + gradient descent) is host-side only.
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

Networks are instantiated entirely through template parameters: no virtual functions, no runtime polymorphism.

### Recurrent networks

- Each LSTM/GRU gate has its own weight matrices (not a shared pre-activation); sharing collapses the output to a constant.
- Call `resetState()` between independent sequences.
- Weight files have a defined order, documented in `cpp/include/nnproperties.hpp` (`NetworkPropertiesFileManager`, `RecurrentNetworkPropertiesFileManager`).
