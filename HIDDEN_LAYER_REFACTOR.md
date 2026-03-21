# Hidden Layer Refactor: Per-Layer Neuron Counts

## Current Constraint

Every hidden layer shares a single `NumberOfNeuronsInHiddenLayers` value. This is baked into 6 structural locations in the codebase:

| Location | What assumes uniformity |
|---|---|
| `HiddenLayerTypeSelector` | Single `InnerHiddenLayerType` for all inner layers |
| `InnerHiddenLayerManager` | Homogeneous byte buffer `[N * sizeof(Layer)]` |
| `GradientsManager` | Formula: `(N-1) * neurons² + neurons` |
| Delta/Gradient/Weight updaters | Runtime loop over `InnerHiddenLayerType*` array |
| `NetworkPropertiesFileManager` | Single `NumberOfHiddenLayerNeurons` in all serialization loops |
| `XavierWeightInitializer` | Uniform size in fan-in/fan-out calculation |

## Proposed Approach: `HiddenLayers<N...>` descriptor

Introduce a variadic template that encodes per-layer sizes as a type:

```cpp
// New: each hidden layer can have a different size
template<size_t... Sizes>
struct HiddenLayers {
    static constexpr size_t Count = sizeof...(Sizes);
    // Sizes accessible via std::index_sequence utilities
};

// Backward-compatible helper: N layers all of size S
template<size_t Count, size_t Size>
using UniformHiddenLayers = /* expands to HiddenLayers<Size, Size, ..., Size> */;
```

**User-facing change — new style:**
```cpp
// Different sizes per layer
typedef MultilayerPerceptron<
    ValueType, 2,
    HiddenLayers<10, 5, 3>,   // 3 hidden layers: 10, 5, 3 neurons
    1, TransferFunctionsType> MyNN;
```

**User-facing change — old style preserved:**
```cpp
// Existing code keeps working with a one-line typedef change,
// or we keep the old 6-param signature as a compatibility alias
typedef MultilayerPerceptron<
    ValueType, 2,
    UniformHiddenLayers<1, 3>,  // 1 hidden layer, 3 neurons each
    1, TransferFunctionsType> XorNN;

// OR: keep the old signature as an alias
template<typename V, size_t I, size_t HL, size_t N, size_t O, typename TF>
using MultilayerPerceptronUniform = MultilayerPerceptron<V, I, UniformHiddenLayers<HL, N>, O, TF>;
```

## Key Internal Changes

### 1. Layer storage — `std::tuple` replaces byte buffer

```cpp
// Before: homogeneous array
unsigned char mInnerHiddenLayersBuffer[N * sizeof(InnerHiddenLayerType)];

// After: heterogeneous tuple generated from the size pack
std::tuple<
    HiddenLayer<Neuron<ValueType, NextLayerSize>, Size>...
> mInnerHiddenLayers;
```

Each layer type is different (different neuron count, different outgoing connection count), so a tuple is the natural compile-time heterogeneous container. No dynamic allocation — same as today.

### 2. Forward/back propagation — compile-time recursion replaces runtime loops

```cpp
// Before: runtime for-loop over pointer
for (size_t i = 1; i < NumberOfInnerHiddenLayers; ++i)
    pInnerHiddenLayers[i].feedForward(pInnerHiddenLayers[i - 1]);

// After: recursive template (or fold expression in C++17)
template<size_t I>
void feedForwardChain() {
    if constexpr (I < NumInnerHiddenLayers) {
        std::get<I>(mLayers).feedForward(std::get<I-1>(mLayers));
        feedForwardChain<I + 1>();
    }
}
```

This pattern applies to all 4 traversal sites (forward prop, delta calc, gradient calc, weight update).

### 3. Gradient count — compile-time sum over adjacent pairs

```cpp
// Before: (N-1) * neurons² + neurons
// After: sum of (size[i] * size[i+1]) for adjacent pairs, computed via parameter pack expansion
template<size_t First, size_t Second, size_t... Rest>
constexpr size_t totalGradients() {
    if constexpr (sizeof...(Rest) == 0)
        return First * Second + Second;  // connections + biases
    else
        return First * Second + Second + totalGradients<Second, Rest...>();
}
```

### 4. Serialization — recursive template over layer sizes

`NetworkPropertiesFileManager` loops would use the same recursive `if constexpr` pattern instead of uniform `for` loops.

### 5. Xavier initialization — per-layer fan-in/fan-out

Each layer gets the correct fan-in (previous layer size) and fan-out (next layer size) from the compile-time size list, which is actually *more correct* than the current uniform approach.

## Impact Assessment

| Who | Impact |
|---|---|
| **Existing users (uniform layers)** | Wrap params in `UniformHiddenLayers<Count, Size>`, OR use the compatibility alias with zero changes |
| **`neuralnet.hpp`** | Major rewrite of `InnerHiddenLayerManager`, all 4 propagation managers, `GradientsManager`. ~800-1000 lines touched |
| **`nnproperties.hpp`** | Moderate rewrite of serialization loops (~200 lines) |
| **`xavier.hpp`** | Small change (~30 lines) |
| **Unit tests / examples** | Typedef changes only — test logic unchanged |
| **C++ standard requirement** | C++14 minimum (for `std::index_sequence`); C++17 preferred (for `if constexpr`) |

## Recommended Migration Strategy

1. **Phase 1**: Add `HiddenLayers<N...>` and `UniformHiddenLayers<Count, Size>` types. Keep the old 6-param `MultilayerPerceptron` signature as an alias to the new form using `UniformHiddenLayers`. **Zero breakage.**

2. **Phase 2**: Rewrite internals (`InnerHiddenLayerManager` → tuple-based, loops → recursive templates). Old alias still works. All existing tests pass without modification.

3. **Phase 3**: Add tests with non-uniform hidden layers (e.g., `HiddenLayers<10, 5, 3>`) to validate the new capability.

This approach gives heterogeneous hidden layers while preserving full backward compatibility through the type alias.
