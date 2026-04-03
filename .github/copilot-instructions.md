# GitHub Copilot Instructions for TinyMind

TinyMind is a **header-only C++ template library** for neural networks and Q-learning, targeting embedded systems with no FPU, GPU, or SIMD requirements. The design follows Andrei Alexandrescu's policy-based design pattern from *Modern C++ Design*.

## Build & Test Commands

```bash
# Build and run all unit tests + examples
make check

# Build and run a specific test suite
cd unit_test/nn && make clean && make && make run
cd unit_test/qformat && make clean && make && make run
cd unit_test/qlearn && make clean && make && make run
cd unit_test/kan && make clean && make && make run

# Run a single test binary directly (after building)
./unit_test/nn/output/nn_unit_test

# Build an example (debug / release)
cd examples/xor && make clean && make         # debug
cd examples/xor && make clean && make release # optimized (-O3)
```

`BOOST_HOME` must be set to the Boost installation path before building unit tests.

Compiler flags used everywhere: `-Wall -Wextra -Werror -Wpedantic`. Debug adds `-ggdb`; release adds `-O3`.

## Architecture

### One .cpp File Rule

`cpp/lookupTables.cpp` is the **only** non-header source file in the library. It contains ~3 MB of pre-computed lookup tables (sigmoid, tanh, exp, log) for fixed-point activation functions. It must be compiled into every binary that uses fixed-point activations.

### Policy-Based Network Configuration

Networks are instantiated entirely through template parameters—no virtual functions, no runtime polymorphism. The canonical pattern:

```cpp
// 1. Choose a value type
typedef tinymind::QValue<8, 8, true> ValueType;   // Q8.8 fixed-point, signed

// 2. Define a random number generator policy
struct MyRNG {
    static ValueType generateRandomWeight() { /* [-1, 1] */ }
};

// 3. Compose transfer functions (hidden activation + output activation)
typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    MyRNG,
    tinymind::ReluActivationPolicy<ValueType>,     // hidden
    tinymind::SigmoidActivationPolicy<ValueType>   // output
> TransferFunctionsType;

// 4. Instantiate the network
typedef tinymind::MultilayerPerceptron<
    ValueType,
    NUMBER_OF_INPUTS,
    NUMBER_OF_HIDDEN_LAYERS,
    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
    NUMBER_OF_OUTPUTS,
    TransferFunctionsType
> NeuralNetworkType;
```

All code lives in the `tinymind` namespace; internal helpers are in `tinymind::detail`.

### Fixed-Point Arithmetic (`qformat.hpp`)

`QValue<IntBits, FracBits, IsSigned, RoundPolicy, SaturationPolicy>` supports Q8.8, Q16.16, Q24.8, Q32.32, and others up to 128-bit, as well as `float` as a drop-in value type. Round and saturation behaviors are swappable policies.

### Memory Management

**No dynamic allocation.** Neurons and connections live in `unsigned char mBuffer[N]` arrays with placement new. Never add `new`/`malloc` to the library.

### Lookup Table Activation Macros

Fixed-point activation functions are gated by preprocessor defines. You must pass the right `-D` flags at compile time and link `lookupTables.cpp`:

| Macro | Activates |
|---|---|
| `TINYMIND_USE_SIGMOID_8_8=1` | 8-bit sigmoid LUT |
| `TINYMIND_USE_TANH_16_16=1` | 16-bit tanh LUT |
| `TINYMIND_USE_EXP_16_16=1` | 16-bit exp LUT |
| `TINYMIND_USE_LOG_16_16=1` | 16-bit log LUT |

8, 16, 32, 64, and 128-bit variants exist for all four functions. See the Makefiles for the exact flags used in each example/test.

`TINYMIND_ENABLE_OSTREAMS=1` enables `<<` output operators (useful for debugging, omit in embedded builds).

## Key Conventions

### All Headers Use `#pragma once`

Do not use `#ifndef` include guards.

### SFINAE for Optional Policies

Optional training features (gradient clipping, weight decay, Adam, etc.) are detected at compile time via SFINAE in `tinymind::detail`. To add an optional policy to a network, add a nested type or static member that the detection machinery looks for—no runtime checks.

### Training Loop Pattern

```cpp
net.feedForward(inputs);   // forward pass (array or initializer_list)
net.trainNetwork(targets); // backprop + weight update
net.getLearnedValues(outputs); // read outputs after feedForward
```

### Weight Serialization (`cpp/include/nnproperties.hpp`)

Weights are stored in a defined order: input→hidden weights, hidden bias, hidden→hidden weights (per additional layer), hidden→output weights, output bias. Text format is one value per line; binary format is raw `sizeof(FullWidthValueType)` bytes. Use `NetworkPropertiesFileManager::readWeightsFromFile` / `writeWeightsToFile`.

### Cross-Compiler Warning Suppression

Use the macros in `include/compiler.h` to suppress specific warnings portably across GCC, Clang, and MSVC:

```cpp
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING("-Wsign-conversion")
// ... code ...
TINYMIND_DISABLE_WARNING_POP
```

### LSTM / GRU Notes

- Each gate has **independent** weight matrices (not a shared pre-activation). Sharing causes collapse to constant output.
- Call `resetState()` between independent sequences.
- Use gradient clipping + Adam + learning rate scheduling for stable LSTM training. See `LSTM.md` for a working reference configuration.

### PyTorch Weight Import Workflow

`examples/pytorch/` demonstrates training in PyTorch and running inference in TinyMind C++.

**Step 1 — Train and export in Python:**

```bash
cd examples/pytorch/xor
python3 xor.py        # trains, then writes input/xor_weights_q16_16.txt
```

The script converts each `float32` weight to a Q16.16 integer (`int(round(x * 65536))`) and writes one integer per line. The value order matches the TinyMind weight file format exactly (see `examples/pytorch/xor/README.md` for the line-by-line layout).

**Step 2 — Load weights in C++:**

The inference-only C++ network sets the 6th template parameter of `MultilayerPerceptron` to `false` (non-trainable) and uses `NullRandomNumberPolicy` because no weight initialization is needed:

```cpp
typedef tinymind::QValue<16, 16, true> ValueType;

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    tinymind::NullRandomNumberPolicy<ValueType>,   // no training → no RNG
    tinymind::ReluActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>
> TransferFunctionsType;

typedef tinymind::MultilayerPerceptron<
    ValueType, 2, 1, 3, 1,
    TransferFunctionsType,
    false                                          // false = inference-only
> NeuralNetworkType;

typedef tinymind::NetworkPropertiesFileManager<NeuralNetworkType> FileManagerType;

std::ifstream f("input/xor_weights_q16_16.txt");
FileManagerType::loadNetworkWeights<ValueType, ValueType>(net, f);
```

**Step 3 — Build and run:**

```bash
make clean && make && cd output && ./xor
```

**Key constraint:** The PyTorch architecture (layer count, neurons per layer, activation functions) must match the C++ template parameters exactly. Q16.16 is the recommended format for imported weights because it retains sufficient precision from `float32`.

### Activation Lookup Table Generator (`apps/activation/`)

`activationTableGenerator.cpp` regenerates all the LUT source files in `cpp/`. Run this if you need to change LUT parameters (table size, value range, delta shift) or add support for a new Q-format.

```bash
cd apps/activation
make clean && make
# Pass the cpp/ directory as the output destination:
./output/activationTableGenerator ../../cpp/
```

This overwrites/regenerates: `activation.hpp`, `sigmoid.hpp`, `tanh.hpp`, `exp.hpp`, `log.hpp`, the per-bit-width `*Values{8,16,32,64,128}Bit.hpp` headers, and `lookupTables.cpp`.

The generator requires C++17 (`std::filesystem`) and links `-lstdc++fs`. The constants `NUMBER_OF_ACTIVATION_TABLE_VALUES`, `MIN_X_TABLE_VALUE`, `MAX_X_TABLE_VALUE`, and `ACTIVATION_DELTA_SHIFT` that control the table shape are defined in `cpp/activation.hpp` and written into the generated files at regeneration time.

After regenerating, rebuild all unit tests and examples with `make check` to verify correctness.

### KAN Notes

- Activation: `phi(x) = w_b * SiLU(x) + w_s * spline(x)` (De Boor B-spline).
- Recommend `SplineDegree=1` (piecewise linear) for fixed-point; Q16.16 for training, Q8.8 for inference.
- See `KAN.md` for memory layout and recommended grid sizes.
