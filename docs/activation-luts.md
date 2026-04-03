---
title: Activation Function Lookup Tables
layout: default
nav_order: 8
---

# Activation Function Lookup Tables

On embedded systems without standard math libraries, functions like `tanh()`, `sigmoid()`, `exp()`, and `log()` are unavailable. TinyMind replaces these with pre-computed **lookup tables (LUTs)** stored as constant arrays in memory. At runtime, activation values are retrieved via table lookup with linear interpolation -- no floating-point math required.

## Why Lookup Tables?

Traditional activation functions require either a hardware FPU or a software floating-point emulation library, both of which may be unavailable or prohibitively expensive on small microcontrollers. LUTs provide:

- **Constant-time evaluation** -- a table index computation and one interpolation, regardless of input value
- **No FPU dependency** -- all arithmetic is integer-only using Q-format fixed-point
- **Predictable memory cost** -- each table is exactly 96 entries, sized by the Q-format's storage type
- **Compile-time selectability** -- preprocessor switches include only the tables you use

## Supported Activation Functions

| Function | LUT Name Prefix | Activation Policy | Derivative |
|---|---|---|---|
| Sigmoid | `SigmoidValuesTableQ` | `SigmoidActivationPolicy<ValueType>` | `f'(x) = f(x) * (1 - f(x))` |
| Tanh | `TanhValuesTableQ` | `TanhActivationPolicy<ValueType>` | `f'(x) = 1 - f(x)^2` |
| Exp | `ExpValuesTableQ` | Used by `SoftmaxActivationPolicy`, `EluActivationPolicy` | N/A (used internally) |
| Log | `LogValuesTableQ` | Available for custom policies | N/A (used internally) |

Additionally, `GeluActivationPolicy` reuses the sigmoid LUT with a 1.702x input scaling factor.

## Table Parameters

All LUTs share the same domain and resolution, defined in `cpp/activation.hpp`:

| Parameter | Value | Meaning |
|---|---|---|
| `NUMBER_OF_ACTIVATION_TABLE_VALUES` | 96 | Number of entries per table |
| `MIN_X_TABLE_VALUE` | -6 | Left edge of the input domain |
| `MAX_X_TABLE_VALUE` | 6 | Right edge of the input domain |
| `ACTIVATION_DELTA_SHIFT` | 3 | Spacing = `1 / 2^3 = 0.125` between sample points |

The input domain [-6, 6] covers the full dynamic range where sigmoid and tanh have meaningful variation. Outside this range, sigmoid saturates at 0 or 1, and tanh at -1 or +1. The 96 entries at 0.125 spacing provide `96 * 0.125 = 12` units of range, exactly covering [-6, +6).

## Q-Format and Table Sizes

Each activation function has tables generated for every valid Q-format split within each supported bit width. The bit widths and corresponding storage types are:

| Total Bits | Storage Type | Example Q-Formats | Table Size (bytes) |
|---|---|---|---|
| 8 | `uint8_t` | Q1.7, Q2.6, Q3.5, ..., Q7.1 | 96 |
| 16 | `uint16_t` | Q1.15, Q8.8, Q12.4, ... | 192 |
| 32 | `uint32_t` | Q16.16, Q24.8, ... | 384 |
| 64 | `uint64_t` | Q32.32, Q48.16, ... | 768 |
| 128 | `uint128_t` | Q64.64, ... | 1,536 |

For a given total bit width `N`, tables are generated for every split from Q1.(N-1) through Q(N-1).1 -- that is, `N-1` tables per activation function per bit width.

## Compile-Time Table Selection

Only the LUTs you need are compiled into your binary. Each table is guarded by a preprocessor macro:

```
TINYMIND_USE_SIGMOID_8_8    // Sigmoid Q8.8
TINYMIND_USE_TANH_8_8       // Tanh Q8.8
TINYMIND_USE_EXP_16_16      // Exp Q16.16
TINYMIND_USE_LOG_24_8       // Log Q24.8
```

The naming pattern is `TINYMIND_USE_{FUNCTION}_{FixedBits}_{FracBits}`. Define these macros in your build system (e.g., `-DTINYMIND_USE_TANH_8_8=1`) to include the corresponding table.

When you instantiate `TanhActivationPolicy<QValue<8,8,true>>`, the compiler resolves the table through a template selector chain:

```
TanhActivationPolicy<ValueType>
  -> TanhValuesTableSelector<8, 8, true>
    -> TanhTableValueSize<8, 8, true>     (template specialization, guarded by #if)
      -> TanhValuesTableQ8_8              (the struct containing the values[] array)
```

If the required `TINYMIND_USE_*` macro is not defined, compilation fails with a clear error at the selector step.

## Runtime Lookup with Linear Interpolation

When an activation function is called at runtime, the `LookupTable<ValueType>::getValue()` method in `cpp/lookupTable.hpp` performs:

1. **Clamp** -- if the input is outside [-6, +6], return the boundary table entry
2. **Index** -- compute the lower table index: `(value - MIN_X) / DELTA_X`
3. **Interpolate** -- linearly interpolate between the lower and upper entries:

```
result = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
```

All arithmetic uses the fixed-point `QValue` type, so no floating-point operations occur at runtime. The interpolation fills in sub-table-step precision, making the 96-entry table behave like a much smoother curve.

## The LUT Generator Application

The `apps/activation/` directory contains the standalone application that generates all LUT source code. This is a build-time code generator -- you run it once to produce the header and source files that ship with the library.

### Building the Generator

```bash
cd apps/activation
make          # debug build
make release  # optimized build
```

### Running the Generator

```bash
./output/activationTableGenerator ../../cpp
```

The single argument is the output directory. The generator produces:

| Output File | Contents |
|---|---|
| `activation.hpp` | Table dimension constants (`NUMBER_OF_ACTIVATION_TABLE_VALUES`, etc.) |
| `sigmoid.hpp`, `tanh.hpp`, `exp.hpp`, `log.hpp` | Template selector structs that map Q-format parameters to the correct table struct |
| `sigmoidValues{8,16,32,64,128}Bit.hpp` | Table struct declarations for each bit width |
| `tanhValues{8,16,32,64,128}Bit.hpp` | (same pattern for tanh) |
| `expValues{8,16,32,64,128}Bit.hpp` | (same pattern for exp) |
| `logValues{8,16,32,64,128}Bit.hpp` | (same pattern for log) |
| `lookupTables.cpp` | All table data -- the `const` arrays with pre-computed hex values (~3 MB) |

### How Values Are Computed

For each Q-format split and each activation function, the generator:

1. Iterates over 96 evenly spaced x-values from -6.0 to +5.875 (step 0.125)
2. Computes the activation using double-precision floating-point (`std::tanh`, `std::exp`, or the sigmoid formula `e^x / (e^x + 1)`)
3. Converts the result to fixed-point by multiplying by `2^FractionalBits`
4. Casts to the appropriate unsigned integer type and writes as a hex literal

For example, `tanh(0.0) = 0.0` in Q8.8 becomes `0x0000`, while `tanh(3.0) = 0.9951` becomes `0x00FE` (254/256 = 0.9922 in Q8.8).

### The Visualization Script

The `lut_parse.py` script parses `lookupTables.cpp` and plots any table using matplotlib:

```bash
cd apps/activation
python lut_parse.py -f tanh -q 8.8     # Plot tanh Q8.8
python lut_parse.py -f sigmoid -q 16.16 # Plot sigmoid Q16.16
```

## Memory Footprint

A typical embedded deployment using Q8.8 with tanh activation needs only one 96-byte table. Here is how table sizes scale:

| Configuration | Tables Needed | Total LUT Memory |
|---|---|---|
| MLP with tanh (Q8.8) | 1 tanh table | 96 bytes |
| MLP with sigmoid (Q8.8) | 1 sigmoid table | 96 bytes |
| MLP with tanh (Q16.16) | 1 tanh table | 192 bytes |
| Softmax output + tanh hidden (Q8.8) | 1 exp + 1 tanh table | 192 bytes |
| GELU activation (Q8.8) | 1 sigmoid table | 96 bytes |
| ELU activation (Q16.16) | 1 exp table | 192 bytes |

Because of the `#if` guards, only the tables matching your defined macros are compiled. A binary using only `TINYMIND_USE_TANH_8_8` pays 96 bytes for activation tables and zero bytes for sigmoid, exp, and log.

## Activation Policies That Use LUTs

Each fixed-point activation policy in `cpp/activationFunctions.hpp` wraps the LUT lookup:

- **`SigmoidActivationPolicy<ValueType>`** -- direct sigmoid table lookup
- **`TanhActivationPolicy<ValueType>`** -- direct tanh table lookup
- **`SoftmaxActivationPolicy<ValueType>`** -- exp table lookup per neuron, then normalization by sum
- **`EluActivationPolicy<ValueType>`** -- exp table lookup for negative inputs, identity for positive
- **`GeluActivationPolicy<ValueType>`** -- sigmoid table lookup with 1.702x input scaling

Policies that do not need LUTs (`LinearActivationPolicy`, `ReluActivationPolicy`, `CappedReluActivationPolicy`, `NullActivationPolicy`) use direct arithmetic and have no table dependency.

## Adding a Custom Q-Format

If you need a Q-format that is not yet enabled:

1. Ensure the total bit width is one of 8, 16, 32, 64, or 128
2. Add the appropriate `-D` flag to your build: `-DTINYMIND_USE_TANH_{Fixed}_{Frac}=1`
3. The table data and selector specialization already exist in `lookupTables.cpp` and the generated headers -- the `#if` guard simply enables them

No code generation step is needed unless you modify the table parameters (domain, resolution, or add a new activation function).
