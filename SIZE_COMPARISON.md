# Network Instance Size Comparison

## NeuralNetwork vs MultilayerPerceptron

Instance sizes in bytes for equivalent network configurations using `double` as the value type.

| Configuration | MultilayerPerceptron | NeuralNetwork |
|---|---|---|
| 1 hidden layer (2→5→1) | 1,000 | 1,000 |
| 2 hidden layers (2→5→5→1) | 2,104 | 2,104 |
| 3 hidden layers (2→5→5→5→1) | 3,208 | 3,208 |
| Large (10→20→20→5) | 25,480 | 25,480 |
| Recurrent/Elman (2→3→1) | 1,048 | 1,048 |
| Non-trainable (2→5→1) | 360 | 360 |

Zero overhead — the chain-based `LayerChain`/`EmptyLayerChain` approach compiles down to the same size as the array-based `InnerHiddenLayerManager`.

## All Architectures: MLP vs LSTM vs KAN

### Using `double` (2→N→1)

Instance sizes in bytes for XOR-class network configurations using `double` as the value type.

| Architecture | Hidden Neurons | Trainable | Non-trainable | Training Overhead |
|---|---|---|---|---|
| MLP (2→5→1) | 5 | 1,000 | 360 | +640 (+178%) |
| Elman RNN (2→3→1) | 3 | 1,048 | — | — |
| LSTM (2→3→1) | 3 | 3,016 | 960 | +2,056 (+214%) |
| KAN (2→5→1, G=5, k=1) | 5 | 4,208 | 1,256 | +2,952 (+235%) |

### Using Q8.8 Fixed-Point (XOR Configuration)

Instance sizes in bytes for XOR network configurations using `QValue<8,8,true>` (signed Q8.8 fixed-point, 2 bytes per value).

| Architecture | Hidden Neurons | Trainable | Non-trainable | Training Overhead |
|---|---|---|---|---|
| MLP (2→3→1) | 3 | 328 | 144 | +184 (+128%) |
| LSTM (2→3→1) | 3 | 952 | 384 | +568 (+148%) |
| KAN (2→5→1, G=5, k=1) | 5 | 1,192 | 416 | +776 (+187%) |

### Relative Size (vs MLP)

| | Trainable | Non-trainable |
|---|---|---|
| **double** | | |
| LSTM / MLP | 3.0x | 2.7x |
| KAN / MLP | 4.2x | 3.5x |
| **Q8.8** | | |
| LSTM / MLP | 2.9x | 2.7x |
| KAN / MLP | 3.6x | 2.9x |

### Why Each Architecture is Larger

- **MLP**: One weight per connection. Minimal storage.
- **LSTM**: 4 gates (input, forget, output, cell) multiply connection weights by 4x, plus recurrent state and cell memory.
- **KAN**: Each edge stores B-spline coefficients (`GridSize + SplineDegree` = 6 per edge with G=5, k=1), plus a base weight and spline weight. Training adds gradient, delta weight, and previous delta weight for every learnable parameter.

All three architectures remain well under 1.2 KB in Q8.8 fixed-point even in trainable form, making them suitable for embedded deployment.
