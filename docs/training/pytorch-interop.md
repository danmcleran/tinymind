---
title: PyTorch Interop
layout: default
parent: Training & Deployment
nav_order: 2
---

# Weight Import/Export and PyTorch Interoperability

tinymind provides weight serialization for all network types (MLP, LSTM, GRU, KAN), enabling a powerful workflow: **train in PyTorch** on a powerful workstation, **export weights** to a text file, and **deploy in tinymind C++** on an embedded device with no training overhead.

## Why This Matters for Embedded

Training a neural network requires 2-3x the memory of inference alone (gradients, delta weights, momentum terms). On a microcontroller with 4-8 KB of RAM, this overhead can be the difference between feasible and impossible. The train-offline-deploy-lean workflow eliminates this entirely:

| Mode | MLP (2->3->1) Q8.8 | LSTM (2->3->1) Q8.8 | GRU (2->3->1) Q8.8 |
|---|---|---|---|
| Trainable | 328 bytes | 952 bytes | 808 bytes |
| Non-trainable (`IsTrainable=false`) | 144 bytes | 384 bytes | 336 bytes |
| Savings | 56% | 60% | 58% |

By setting `IsTrainable=false`, the compiler strips all training infrastructure -- backpropagation code, gradient storage, optimizer state -- and produces a minimal inference-only binary.

This approach also provides **on-device privacy** (sensor data never leaves the device), **zero-latency inference** (no network round-trip), and **battery efficiency** (no radio transmission).

# Weight File Managers

tinymind provides three file manager templates, one for each network family:

| File Manager | Network Types | Header |
|---|---|---|
| `NetworkPropertiesFileManager<NNType>` | MLP, NeuralNetwork | `nnproperties.hpp` |
| `RecurrentNetworkPropertiesFileManager<NNType>` | LSTM, GRU, Elman | `nnproperties.hpp` |
| `KanNetworkPropertiesFileManager<KanType>` | KAN | `nnproperties.hpp` |

## Common API

```cpp
// Save weights to text file (one value per line)
std::ofstream outFile("weights.txt");
FileManager::storeNetworkWeights(network, outFile);

// Save weights to binary file
FileManager::storeNetworkWeights(network, "weights.bin");

// Load weights from text file
std::ifstream inFile("weights.txt");
FileManager::template loadNetworkWeights<SourceType, DestType>(network, inFile);
```

# Weight File Formats

## Feed-Forward Networks (MLP)

Values are stored one per line in this order:

1. **Input-to-first-hidden weights**: `NumberOfInputs * NumberOfHiddenNeurons`
2. **Input layer bias weights**: `NumberOfHiddenNeurons`
3. **Hidden-to-hidden weights** (when `NumberOfHiddenLayers > 1`): `NumberOfHiddenNeurons^2 + NumberOfHiddenNeurons` per layer
4. **Last-hidden-to-output weights**: `NumberOfHiddenNeurons * NumberOfOutputs`
5. **Output layer bias weights**: `NumberOfOutputs`

## Recurrent Networks (LSTM, GRU)

1. **Input-to-hidden weights (gated)**: `NumberOfInputs * NumberOfHiddenNeurons * GateMultiplier`
2. **Input layer bias weights (gated)**: `NumberOfHiddenNeurons * GateMultiplier`
3. **Recurrent-to-hidden weights (gated)**: `NumberOfHiddenNeurons^2 * GateMultiplier`
4. **Last-hidden-to-output weights** (not gated): `NumberOfHiddenNeurons * NumberOfOutputs`
5. **Output layer bias weights**: `NumberOfOutputs`

`GateMultiplier` is 4 for LSTM and 3 for GRU.

## Value Encoding

- **Fixed-point**: Raw integer representation scaled by `2^FractionalBits`. For example, 1.5 in Q16.16 is stored as `98304` (= 1.5 * 65536).
- **Floating-point**: Standard decimal string representation.

# PyTorch Export: MLP

Source code: [examples/pytorch/xor/xor.py](https://github.com/danmcleran/tinymind/blob/master/examples/pytorch/xor/xor.py)

## Q16.16 Conversion Function

```python
def float_to_q16_16(x: float) -> int:
    """Convert a Python float to signed Q16.16 integer representation."""
    val = int(round(x * (1 << 16)))
    if val < -2**31:
        val = -2**31
    elif val > 2**31 - 1:
        val = 2**31 - 1
    return val
```

## Export Function

The export function writes weights in the exact order that `NetworkPropertiesFileManager::loadNetworkWeights` expects:

```python
def save_to_tinymind_format(self, path: str) -> None:
    from collections import OrderedDict
    data = OrderedDict()

    # 1. Input -> hidden weights (transposed: PyTorch stores [out, in])
    rows, cols = self.fc1.weight.T.shape
    for i in range(rows):
        for j in range(cols):
            data[f'Input{i}{j}Weight'] = float_to_q16_16(self.fc1.weight.T[i, j].item())

    # 2. Input bias -> hidden
    for j in range(len(self.fc1.bias)):
        data[f'InputBias0{j}Weight'] = float_to_q16_16(self.fc1.bias[j].item())

    # 3. Hidden -> output weights
    rows, cols = self.fc2.weight.T.shape
    for i in range(rows):
        for j in range(cols):
            data[f'Hidden0{i}{j}Weight'] = float_to_q16_16(self.fc2.weight.T[i, j].item())

    # 4. Hidden bias -> output
    for j in range(len(self.fc2.bias)):
        data[f'Hidden0Bias{j}Weight'] = float_to_q16_16(self.fc2.bias[j].item())

    with open(path, 'w') as f:
        f.write('\n'.join(str(v) for v in data.values()) + '\n')
```

# C++ Import: MLP Inference

Source code: [examples/pytorch/xor/xor.cpp](https://github.com/danmcleran/tinymind/blob/master/examples/pytorch/xor/xor.cpp)

```cpp
typedef tinymind::QValue<16, 16, true> ValueType;

// Non-trainable network (inference only) -- saves memory
typedef tinymind::FixedPointTransferFunctions<ValueType,
    tinymind::NullRandomNumberPolicy<ValueType>,
    tinymind::ReluActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>> TransferFunctionsType;

typedef tinymind::MultilayerPerceptron<ValueType, 2, 1, 3, 1,
    TransferFunctionsType, false> NeuralNetworkType; // false = non-trainable

typedef tinymind::NetworkPropertiesFileManager<NeuralNetworkType> FileManager;

NeuralNetworkType testNeuralNet;

// Load PyTorch-exported weights
std::ifstream weightsFile("../input/xor_weights_q16_16.txt");
FileManager::template loadNetworkWeights<ValueType, ValueType>(testNeuralNet, weightsFile);

// Run inference
ValueType values[2], learnedValues[1];
values[0] = ValueType(1, 0); // 1.0
values[1] = ValueType(0, 0); // 0.0
testNeuralNet.feedForward(&values[0]);
testNeuralNet.getLearnedValues(&learnedValues[0]);
// learnedValues[0] should be close to 1.0 (XOR of 1,0)
```

# PyTorch Export: GRU

Source code: [examples/pytorch/gru/](https://github.com/danmcleran/tinymind/tree/master/examples/pytorch/gru) (`gru_export.py`, plus `gru_parity.cpp` and `test_gru_parity.py`, which check the result)

PyTorch and tinymind write the same GRU in three different ways, and an export has to account for all three:

1. **Gate order.** PyTorch stores the gates as r (reset), z (update), n (new/candidate); tinymind uses z (update), r (reset), n (candidate).
2. **Update-gate convention.** PyTorch computes `h_t = (1 - z) * n + z * h_{t-1}`; tinymind computes `h_t = (1 - z') * h_{t-1} + z' * n`. So tinymind's `z'` is PyTorch's `1 - z = sigmoid(-a)`, and every update-gate weight and bias is negated on export.
3. **The candidate's recurrent bias.** PyTorch's candidate is `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`, so the reset gate scales `b_hn`. tinymind has one bias per gate, added outside the reset gate, so a nonzero `b_hn` has no equivalent. The example's `GRUNet` sets `b_hn` to zero and masks its gradient so it stays zero through training, and the exporter refuses a model whose `b_hn` is not zero.

```python
class GRUNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        # b_hn (the last third of bias_hh_l0) has no slot in tinymind: keep it at 0.
        H = hidden_size
        with torch.no_grad():
            self.gru.bias_hh_l0[2 * H:] = 0.0
        mask = torch.ones(3 * H)
        mask[2 * H:] = 0.0
        self.gru.bias_hh_l0.register_hook(lambda grad: grad * mask)

def export_gru_weights(model, path, use_q16_16=True):
    check_exportable(model)   # raises ValueError if b_hn != 0
    H = model.hidden_size
    convert = float_to_q16_16 if use_q16_16 else (lambda x: repr(x))

    gate_reorder = [1, 0, 2]        # PyTorch [r, z, n] -> tinymind [z, r, n]
    gate_sign = [-1.0, 1.0, 1.0]    # tinymind z' = 1 - z: negate the update gate

    w_ih = model.gru.weight_ih_l0.detach().double().numpy()  # [3H, I]
    w_hh = model.gru.weight_hh_l0.detach().double().numpy()  # [3H, H]
    b_ih = model.gru.bias_ih_l0.detach().double().numpy()
    b_hh = model.gru.bias_hh_l0.detach().double().numpy()
    w_out = model.fc.weight.detach().double().numpy()
    b_out = model.fc.bias.detach().double().numpy()

    I = w_ih.shape[1]
    values = []

    # 1. Input-to-hidden weights (gated, reordered, update gate negated)
    for i in range(I):
        for h in range(H):
            for g in range(3):
                pg = gate_reorder[g]
                values.append(convert(gate_sign[g] * float(w_ih[pg * H + h, i])))

    # 2. One bias per gate: b_ih + b_hh (b_hn is zero)
    for h in range(H):
        for g in range(3):
            pg = gate_reorder[g]
            values.append(convert(gate_sign[g] * float(b_ih[pg * H + h] + b_hh[pg * H + h])))

    # 3. Recurrent-to-hidden weights (gated, reordered, update gate negated)
    for r in range(H):
        for h in range(H):
            for g in range(3):
                pg = gate_reorder[g]
                values.append(convert(gate_sign[g] * float(w_hh[pg * H + h, r])))

    # 4. Hidden-to-output weights (not gated)
    for h in range(H):
        for o in range(w_out.shape[0]):
            values.append(convert(float(w_out[o, h])))

    # 5. Output bias
    for o in range(len(b_out)):
        values.append(convert(float(b_out[o])))

    with open(path, 'w') as f:
        for v in values:
            f.write(f"{v}\n")
```

Key details:
- Weight matrices are transposed (PyTorch uses `[out_features, in_features]` layout).
- `make parity PYTHON=<python with torch>` in `examples/pytorch/gru` trains the model, exports it, loads the file into tinymind's `GruNeuralNetwork` and compares outputs: double matches PyTorch to about 5e-13, Q16.16 to about 0.003, and the XOR model classifies 4/4 in both. Run it after changing either side of the mapping. Before the update-gate and `b_hn` handling, the same XOR model scored 4/4 in PyTorch and 1/4 in tinymind.

# Workflow Summary

1. **Train** your network in PyTorch with full floating-point precision, GPU acceleration, and the full PyTorch ecosystem.
2. **Export** weights using the provided Python scripts, converting to Q-format integer representation if deploying with fixed-point.
3. **Load** weights in C++ using the appropriate `FileManager::loadNetworkWeights()`.
4. **Deploy** as a non-trainable (`IsTrainable=false`) network for minimum memory footprint.

This workflow gives you the best of both worlds: state-of-the-art training infrastructure from PyTorch and ultra-efficient inference from tinymind C++.

# Int8 Affine Quantization Path

The flow above uses tinymind's existing single-`ValueType` Q-format pipeline -- the same network template trains in float on the host and runs in Q8.8 / Q16.16 on the MCU. For a TFLite / CMSIS-NN style **post-training int8** deployment, tinymind ships a parallel `Q*` layer family ([`QDense`, `QConv2D`, `QDepthwiseConv2D`, `QPointwiseConv2D`, `QPool2D`]({{ site.baseurl }}/architectures/int8-quantization)) plus host-side calibration helpers in `cpp/include/qcalibration.hpp`. Weights and activations are int8, accumulators are int32, and a per-layer integer Requantizer rescales between layers without any floating-point math at runtime.

The end-to-end PyTorch -> int8 deployment example lives at [`examples/pytorch_quant/xor/`](https://github.com/danmcleran/tinymind/tree/master/examples/pytorch_quant/xor) -- see [PyTorch -> tinymind int8 (XOR)]({{ site.baseurl }}/getting-started/pytorch-quant-xor) for the walkthrough. The shape of that flow is the same as above (train, export, load, deploy), but the export step additionally fits per-tensor activation `(scale, zero_point)` from observed ranges and emits int8 weights plus int32 biases instead of Q-format integers.
