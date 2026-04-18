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

### Examples (`examples/`)

- **`xor/`** — XOR gate learned by a small neural network; includes a Python plotting script
- **`maze/`** and **`dqn_maze/`** — Maze solving via Q-learning and deep Q-networks
- **`pytorch/`** — Exports weights from a PyTorch model and imports them into a TinyMind C++ network for inference
- **`kws_cortex_m/`** — Keyword-spotting-style pipeline built from `Conv2D` → `MaxPool2D` → `DepthwiseConv2D` → `PointwiseConv2D` → `GlobalAvgPool2D` → dense, with a CSV cycles/bytes report from the bench harness. Host runner; includes a `port_stub.hpp` sketch for porting to a Cortex-M target.

### Apps (`apps/activation/`)

Standalone tool that generates the lookup table values used in `lookupTables.cpp`.

### Benchmark harness (`cpp/include/bench/`)

- **`platform.hpp`** — `bench::readCycleCounter()` (DWT CYCCNT on Cortex-M, `std::chrono` host fallback) and `bench::paintStack` / `bench::stackHighWater` for MCU stack watermarking. Enable the MCU path with `-DTINYMIND_BENCH_CORTEX_M`.
- **`report.hpp`** — `bench::LayerStat` CSV row type, `bench::writeHeader` / `bench::writeRow` for any sink that supports `operator<<`, and a `bench::ScopedTimer` for per-layer measurements. See `examples/kws_cortex_m/` for an end-to-end use.
