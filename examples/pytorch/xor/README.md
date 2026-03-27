# PyTorch XOR Weight Export Example

This example demonstrates training a neural network in PyTorch, exporting the
learned weights to a text file, and loading them into a TinyMind C++ network
for inference.

## Workflow

1. **Train in PyTorch** (`xor.py`): Trains a 2-3-1 network on the XOR problem
   and exports the weights as Q16.16 fixed-point integers.
2. **Run inference in C++** (`xor.cpp`): Loads the exported weights into an
   untrainable TinyMind `MultilayerPerceptron` and runs 1000 random XOR
   evaluations.

## Weight File Format

The weight file is a plain text file with **one value per line**. Values are
ordered to match the layer structure of the network, from input to output.

### Value Encoding

| Network value type | File representation |
|--------------------|---------------------|
| Fixed-point (e.g. Q16.16) | Raw integer scaled by 2^(fractional bits). For Q16.16, the float value `2.5` is stored as `163840` (= 2.5 * 65536). |
| Floating-point (`float` / `double`) | Standard decimal string (e.g. `2.5`). |

### Value Order

The values appear in the following order:

#### 1. Input-to-first-hidden-layer weights

```
For each input neuron i (0 .. I-1):
  For each hidden neuron h (0 .. H-1):
    weight[i][h]
```

Count: `I * H`

#### 2. Input layer bias weights

```
For each hidden neuron h (0 .. H-1):
  bias[h]
```

Count: `H`

#### 3. Hidden-to-hidden layer weights (only when L > 1)

```
For each inner hidden layer l (0 .. L-2):
  For each neuron h in layer l (0 .. H-1):
    For each neuron h1 in layer l+1 (0 .. H-1):
      weight[l][h][h1]
  Bias weights for layer l:
    For each neuron h1 in layer l+1 (0 .. H-1):
      bias[l][h1]
```

Count per layer: `H^2 + H`

#### 4. Last-hidden-layer-to-output weights

```
For each hidden neuron h (0 .. H-1):
  For each output neuron o (0 .. O-1):
    weight[h][o]
```

Count: `H * O`

#### 5. Output layer bias weights

```
For each output neuron o (0 .. O-1):
  bias[o]
```

Count: `O`

### Total Value Count

| Configuration | Formula |
|---------------|---------|
| Single hidden layer (L=1) | `I*H + H + H*O + O` |
| Multiple hidden layers | `I*H + H + (L-1)*(H^2 + H) + H*O + O` |

Where: `I` = inputs, `H` = neurons per hidden layer, `O` = outputs, `L` = number
of hidden layers.

### Concrete Example (this XOR network)

Network: 2 inputs, 1 hidden layer with 3 neurons, 1 output (Q16.16).

```
Line  1: weight input[0] -> hidden[0]
Line  2: weight input[0] -> hidden[1]
Line  3: weight input[0] -> hidden[2]
Line  4: weight input[1] -> hidden[0]
Line  5: weight input[1] -> hidden[1]
Line  6: weight input[1] -> hidden[2]
Line  7: bias -> hidden[0]
Line  8: bias -> hidden[1]
Line  9: bias -> hidden[2]
Line 10: weight hidden[0] -> output[0]
Line 11: weight hidden[1] -> output[0]
Line 12: weight hidden[2] -> output[0]
Line 13: bias -> output[0]
```

Total: `2*3 + 3 + 3*1 + 1 = 13` values.

## Binary Format

The binary format (`storeNetworkWeights(neuralNetwork, filePath)`) uses the same
value order. Each value is written as a `FullWidthValueType` in platform-native
byte order, with `sizeof(FullWidthValueType)` bytes per value.

## Building and Running

```bash
# Export weights from PyTorch
python3 xor.py

# Build and run C++ inference
make clean && make && cd output && ./xor
```
