# Kolmogorov-Arnold Network (KAN) Implementation Plan for TinyMind

## Context

Kolmogorov-Arnold Networks (KANs) are a fundamentally different neural network architecture based on the Kolmogorov-Arnold representation theorem. Unlike MLPs which use fixed activation functions on nodes with learnable scalar weights on edges, KANs place **learnable univariate functions (B-splines) on edges** and use simple summation at nodes. This yields better parameter efficiency and interpretability, especially for regression and scientific computing tasks.

Adding KAN support to TinyMind extends its capabilities while maintaining the library's core principles: fixed-point support via QValue, no FPU/GPU required, header-only, compile-time sizing, zero dynamic allocation, and policy-based template design.

No published fixed-point KAN implementations exist, making this novel work.

---

## Mathematical Foundation

**KAN edge function:** `phi(x) = w_b * SiLU(x) + w_s * spline(x)`

Where:
- `SiLU(x) = x * sigmoid(x)` — residual activation (reuses existing sigmoid lookup tables)
- `spline(x) = sum_i(c_i * B_{i,k}(x))` — B-spline with learnable coefficients `c_i`
- `B_{i,k}(x)` — B-spline basis functions of degree `k`, evaluated via De Boor's algorithm
- `w_b`, `w_s` — learnable base and spline scaling weights

**KAN node output:** `y_j = sum_i(phi_{i,j}(x_i))` — pure summation, no node activation

**De Boor's algorithm** uses only add, subtract, multiply, and division by knot differences (which are constants for uniform grids and can be pre-computed as reciprocals). This makes it compatible with QValue fixed-point arithmetic.

---

## New Files to Create

| File | Purpose |
|------|---------|
| `cpp/bspline.hpp` | B-spline evaluation engine (De Boor algorithm, knot vectors) |
| `cpp/kan.hpp` | KAN connections, neurons, layers, and top-level network class |
| `cpp/kanTransferFunctions.hpp` | KAN-specific transfer functions policy, SiLU activation |
| `examples/kan_xor/kan_xornet.h` | XOR example network definition |
| `examples/kan_xor/kan_xor.cpp` | XOR example training program |
| `examples/kan_xor/Makefile` | Build for KAN XOR example |
| `unit_test/kan/kan_unit_test.cpp` | Boost.Test unit tests for KAN |
| `unit_test/kan/Makefile` | Build for KAN unit tests |

**No existing files need modification** (except top-level `Makefile` to add KAN targets to `check`).

---

## Implementation Phases

### Phase 1: B-Spline Engine (`cpp/bspline.hpp`)

**UniformKnotVector** — stores a uniform knot vector as a compile-time-sized array:
```cpp
template<typename ValueType, size_t GridSize, size_t SplineDegree>
struct UniformKnotVector {
    static const size_t NumberOfKnots = GridSize + 2 * SplineDegree + 1;
    static const size_t NumberOfBasisFunctions = GridSize + SplineDegree;
    ValueType knots[NumberOfKnots];
    void initialize(const ValueType& gridMin, const ValueType& gridMax);
};
```
Knot computation: `knot[i] = gridMin + (i - SplineDegree) * spacing` where `spacing = (gridMax - gridMin) / GridSize`. For fixed-point, `GridSize` is a compile-time constant so the division is by a known integer.

**DeBoorEvaluator** — non-recursive triangular-table implementation of De Boor's algorithm:
```cpp
template<typename ValueType, size_t SplineDegree>
struct DeBoorEvaluator {
    static ValueType evaluateSpline(const ValueType* coefficients,
                                    const ValueType* knots,
                                    size_t numberOfBasisFunctions,
                                    const ValueType& x);
    static ValueType evaluateSplineDerivative(const ValueType* coefficients,
                                              const ValueType* knots,
                                              size_t numberOfBasisFunctions,
                                              const ValueType& x);
};
```

**Specialization for `SplineDegree=1`** — reduces to piecewise linear interpolation, reusing `tinymind::linearInterpolation` from `cpp/interpolate.hpp`. This is the recommended mode for fixed-point targets.

The working array for De Boor is `SplineDegree + 1` values on the stack (at most 4 for cubic). No dynamic allocation.

For uniform grids, divisions by knot differences are divisions by multiples of `spacing`, which is constant. Pre-compute `reciprocalSpacing = 1/spacing` at initialization and convert divisions to multiplications at evaluation time.

### Phase 2: KAN Connections (`cpp/kan.hpp`)

**KanConnection** — replaces MLP's single-weight `Connection` with spline coefficients:
```cpp
template<typename ValueType, size_t GridSize, size_t SplineDegree>
struct KanConnection {
    static const size_t NumberOfCoefficients = GridSize + SplineDegree;
    ValueType mCoefficients[NumberOfCoefficients];
    ValueType mBaseWeight;    // w_b for SiLU residual
    ValueType mSplineWeight;  // w_s for spline scaling
    // evaluate(x, knots) -> w_b * SiLU(x) + w_s * spline(x)
};
```

**TrainableKanConnection** — extends with gradient/delta storage per coefficient:
```cpp
template<typename ValueType, size_t GridSize, size_t SplineDegree>
struct TrainableKanConnection : KanConnection<...> {
    ValueType mCoefficientGradients[NumberOfCoefficients];
    ValueType mCoefficientDeltaWeights[NumberOfCoefficients];
    ValueType mCoefficientPreviousDeltaWeights[NumberOfCoefficients];
    ValueType mBaseWeightGradient, mBaseWeightDeltaWeight, mBaseWeightPreviousDeltaWeight;
    ValueType mSplineWeightGradient, mSplineWeightDeltaWeight, mSplineWeightPreviousDeltaWeight;
};
```

**KanConnectionTypeSelector** — selects trainable vs non-trainable based on `IsTrainable` bool, following the existing `ConnectionTypeSelector` pattern.

**Memory per trainable connection** (GridSize=5, SplineDegree=3, Q8.8 = 2 bytes): `(8 coeffs * 3 arrays + 3 * 3 scalars) * 2 bytes = 66 bytes`. A [2,5,1] KAN has 15 connections = ~1 KB. Well within embedded stack limits.

### Phase 3: SiLU Activation & Transfer Functions (`cpp/kanTransferFunctions.hpp`)

**SiLUActivationPolicy** — SiLU(x) = x * sigmoid(x):
```cpp
template<typename ValueType>
struct SiLUActivationPolicy {
    static ValueType activationFunction(const ValueType& value) {
        return value * SigmoidActivationPolicy<ValueType>::activationFunction(value);
    }
    static ValueType activationFunctionDerivative(const ValueType& value) {
        // sig(x) + x * sig(x) * (1 - sig(x))
        const ValueType sig = SigmoidActivationPolicy<ValueType>::activationFunction(value);
        return sig + value * sig * (Constants<ValueType>::one() - sig);
    }
};
```
This reuses the existing sigmoid lookup tables — no new lookup tables needed for SiLU.

**KanTransferFunctions** — mirrors `FixedPointTransferFunctions` but removes hidden/output activation (KAN nodes are just sums):
```cpp
template<typename ValueType,
         class RandomNumberGeneratorPolicy,
         size_t GridSize, size_t SplineDegree,
         unsigned NumberOfOutputNeurons = 1,
         class NetworkInitializationPolicy = DefaultNetworkInitializer<ValueType>,
         class ErrorCalculatorPolicy = MeanSquaredErrorCalculator<ValueType, NumberOfOutputNeurons>,
         class ZeroTolerancePolicy = ZeroToleranceCalculator<ValueType>>
struct KanTransferFunctions {
    // calculateError(), generateRandomWeight(), initialLearningRate(), etc.
    // Same delegation pattern as FixedPointTransferFunctions
    // No hiddenNeuronActivationFunction/outputNeuronActivationFunction
};
```

### Phase 4: KAN Neurons and Layers (`cpp/kan.hpp`)

**KAN neurons** — each neuron owns outgoing KanConnections. Output is set by the layer (sum of incoming connection evaluations). Follows the existing `Neuron`/`TrainableNeuron` pattern with `unsigned char mOutgoingConnectionsBuffer[]` and placement new.

- `KanInputLayerNeuron` / `TrainableKanInputLayerNeuron` — latches input, has outgoing KAN connections
- `KanHiddenLayerNeuron` / `TrainableKanHiddenLayerNeuron` — stores summed output, has outgoing KAN connections
- `KanOutputLayerNeuron` — stores summed output, no outgoing connections, has node delta for training

**KAN layers** — each layer owns a shared `UniformKnotVector`:

- `KanInputLayer<NeuronType, NumNeurons, GridSize, SplineDegree>` — latches inputs, initializes knots
- `KanHiddenLayer<NeuronType, NumNeurons, GridSize, SplineDegree>` — feedForward sums incoming spline evaluations
- `KanOutputLayer<NeuronType, NumNeurons>` — feedForward sums incoming spline evaluations (no outgoing connections)

**Forward pass for hidden/output layer:**
```
for each neuron j in this layer:
    sum = 0
    for each neuron i in previous layer:
        sum += previousLayer.neuron[i].evaluateConnectionOutput(j, previousLayer.knotVector)
    neuron[j].setOutputValue(sum)
```

### Phase 5: KAN Training Infrastructure (`cpp/kan.hpp`)

**Backpropagation gradients for KAN edge** `phi(x) = w_b * SiLU(x) + w_s * spline(x)`:
- `d(phi)/d(c_i) = w_s * B_{i,k}(x)` — gradient w.r.t. each spline coefficient
- `d(phi)/d(w_b) = SiLU(x)` — gradient w.r.t. base weight
- `d(phi)/d(w_s) = spline(x)` — gradient w.r.t. spline weight
- `d(phi)/d(x) = w_b * SiLU'(x) + w_s * spline'(x)` — needed for chain rule to propagate deltas backward

**KanNodeDeltasCalculator:**
- Output layer: `delta_j = target_j - output_j` (no activation derivative since output is linear sum)
- Hidden layers: `delta_j = sum_k(d(phi_{j,k})/d(x_j) * delta_k)` using spline derivatives

**KanGradientsCalculator:** Computes and accumulates gradients for all coefficients, base weights, and spline weights on each connection.

**KanBackPropagationPolicy:** Weight update using learning rate, momentum, and acceleration (same formula as MLP): `new_param = old_param + lr * gradient + momentum * delta + acceleration * prev_delta`

### Phase 6: Top-Level Network Class (`cpp/kan.hpp`)

```cpp
template<typename ValueType,
         size_t NumberOfInputs,
         size_t NumberOfHiddenLayers,
         size_t NumberOfNeuronsInHiddenLayers,
         size_t NumberOfOutputs,
         typename TransferFunctionsPolicy,
         bool IsTrainable = true,
         size_t BatchSize = 1,
         size_t GridSize = 5,
         size_t SplineDegree = 3>
class KolmogorovArnoldNetwork {
public:
    void feedForward(ValueType const* const values);
    ValueType calculateError(ValueType const* const targetValues);
    void trainNetwork(ValueType const* const targetValues);
    void getLearnedValues(ValueType* output) const;
    // Same API as MultilayerPerceptron
};
```

Uses `InnerKanHiddenLayerManager` (mirroring the existing `InnerHiddenLayerManager`) for multi-hidden-layer support.

### Phase 7: XOR Example (`examples/kan_xor/`)

```cpp
// kan_xornet.h - user-facing configuration
typedef tinymind::QValue<8, 8, true> ValueType;
typedef tinymind::KanTransferFunctions<ValueType, RandomNumberGenerator,
    5, 1, 1> TransferFunctionsType;  // GridSize=5, SplineDegree=1
typedef tinymind::KolmogorovArnoldNetwork<ValueType, 2, 1, 5, 1,
    TransferFunctionsType> KanNetworkType;
```

Training loop identical to the MLP XOR example: `feedForward`, `calculateError`, `trainNetwork`, `getLearnedValues`.

### Phase 8: Unit Tests (`unit_test/kan/`)

1. B-spline basis function correctness (known values for k=1,2,3)
2. Uniform knot vector generation
3. KanConnection evaluation with known coefficients
4. KAN forward pass on small [2,3,1] network with pre-set coefficients
5. Gradient verification via finite differences
6. XOR training convergence (fixed-point Q8.8)
7. XOR training convergence (float/double)
8. Inference-only mode (non-trainable KAN)
9. Piecewise linear (k=1) specialization correctness

---

## Key Existing Code to Reuse

| Existing | Used For |
|----------|----------|
| `cpp/qformat.hpp` — `QValue<N,F,S>` | All fixed-point arithmetic |
| `cpp/interpolate.hpp` — `linearInterpolation()` | SplineDegree=1 evaluation |
| `cpp/activationFunctions.hpp` — `SigmoidActivationPolicy` | SiLU computation (sigmoid lookup) |
| `cpp/constants.hpp` — `Constants<ValueType>` | Zero, one, negativeOne constants |
| `cpp/error.hpp` — `MeanSquaredErrorCalculator` | Error calculation |
| `cpp/nninit.hpp` — `DefaultNetworkInitializer` | Learning rate, momentum defaults |
| `cpp/zeroTolerance.hpp` — `ZeroToleranceCalculator` | Zero tolerance checks |
| `cpp/neuralnet.hpp` — buffer/placement-new pattern | Neuron/connection storage idiom |

---

## Fixed-Point Considerations

- **SplineDegree=1 recommended** for fixed-point: reduces to piecewise linear interpolation, avoids accumulated rounding errors from De Boor iterations
- **Uniform grid spacing** makes all knot differences equal, so division becomes multiplication by a pre-computed reciprocal
- **SiLU** reuses existing sigmoid lookup tables — no new tables needed
- **Overflow risk** in `x * sigmoid(x)`: sigmoid is bounded [0,1], so the product magnitude <= |x|. Safe for typical input ranges.
- **Q16.16 recommended** for KAN training (wider fractional bits for gradient precision); Q8.8 viable for inference
- **Grid range [-1, 1]** default; inputs should be normalized to this range

---

## Verification Plan

1. **B-spline unit tests**: Compare evaluator output against hand-computed values for k=1,2,3
2. **Gradient tests**: Finite-difference verification of analytical gradients
3. **XOR convergence**: Train KAN on XOR problem, verify error converges below threshold (both fixed-point and float)
4. **Inference consistency**: Pre-set coefficients, verify forward pass output matches expected values
5. **Build integration**: `make check` runs all KAN tests alongside existing tests

---

## Implementation Summary

**Branch:** `kan_implementation`

### New Files (2,716 lines added)

| File | Purpose |
|------|---------|
| `cpp/bspline.hpp` | B-spline engine — De Boor's algorithm, uniform knot vectors, k=0/1/2/3+ support |
| `cpp/kan.hpp` | Core KAN — connections, neurons, layers, training (backprop), `KolmogorovArnoldNetwork` class |
| `cpp/kanTransferFunctions.hpp` | SiLU activation policy, KAN transfer functions |
| `examples/kan_xor/` | XOR example with Q8.8 fixed-point |
| `unit_test/kan/` | 15 Boost.Test cases — all passing |

### Key Design Decisions

- **Parallel class** to `MultilayerPerceptron`, same user-facing API
- **Same template policy patterns** as existing codebase (type selectors, placement new buffers, no dynamic allocation)
- **SplineDegree=1 specialization** reduces to `linearInterpolation()` — ideal for fixed-point
- **SiLU reuses existing sigmoid lookup tables** — zero new tables needed
- **Training verified** with double precision (XOR converges to error 0.00025)
- **All existing `make check` tests continue to pass**
