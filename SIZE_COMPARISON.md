# Network Instance Size Comparison

## NeuralNetwork vs MultilayerPerceptron

Instance sizes in bytes for equivalent network configurations using `double` as the value type.

| Configuration | MultilayerPerceptron | NeuralNetwork |
|---|---|---|
| 1 hidden layer (2→5→1) | 1,008 | 1,008 |
| 2 hidden layers (2→5→5→1) | 2,112 | 2,112 |
| 3 hidden layers (2→5→5→5→1) | 3,216 | 3,216 |
| Recurrent/Elman (2→3→1) | 1,056 | 1,056 |
| Non-trainable (2→5→1) | 360 | 360 |

Zero overhead — the chain-based `LayerChain`/`EmptyLayerChain` approach compiles down to the same size as the array-based `InnerHiddenLayerManager`.

## All Architectures: MLP vs LSTM vs GRU vs KAN

### Using `double` (2→N→1)

Instance sizes in bytes for XOR-class network configurations using `double` as the value type.

| Architecture | Hidden Neurons | Trainable | Non-trainable | Training Overhead |
|---|---|---|---|---|
| MLP (2→5→1) | 5 | 1,008 | 360 | +648 (+180%) |
| Elman RNN (2→3→1) | 3 | 1,056 | — | — |
| LSTM (2→3→1) | 3 | 3,024 | 960 | +2,064 (+215%) |
| GRU (2→3→1) | 3 | 2,400 | 792 | +1,608 (+203%) |
| KAN (2→5→1, G=5, k=1) | 5 | 4,208 | 1,256 | +2,952 (+235%) |

### Using Q8.8 Fixed-Point (XOR Configuration)

Instance sizes in bytes for XOR network configurations using `QValue<8,8,true>` (signed Q8.8 fixed-point, 2 bytes per value).

| Architecture | Hidden Neurons | Trainable | Non-trainable | Training Overhead |
|---|---|---|---|---|
| MLP (2→3→1) | 3 | 328 | 144 | +184 (+128%) |
| LSTM (2→3→1) | 3 | 952 | 384 | +568 (+148%) |
| GRU (2→3→1) | 3 | 808 | 336 | +472 (+140%) |
| KAN (2→5→1, G=5, k=1) | 5 | 1,192 | 416 | +776 (+187%) |

### Relative Size (vs MLP)

| | Trainable | Non-trainable |
|---|---|---|
| **double** | | |
| LSTM / MLP | 3.0x | 2.7x |
| GRU / MLP | 2.4x | 2.2x |
| KAN / MLP | 4.2x | 3.5x |
| **Q8.8** | | |
| LSTM / MLP | 2.9x | 2.7x |
| GRU / MLP | 2.5x | 2.3x |
| KAN / MLP | 3.6x | 2.9x |

### GRU vs LSTM

| | Trainable | Non-trainable |
|---|---|---|
| **double**: GRU / LSTM | 79% | 83% |
| **Q8.8**: GRU / LSTM | 85% | 88% |

GRU uses 3 gates (update, reset, candidate) versus LSTM's 4 gates (input, forget, output, cell candidate), saving ~15-21% memory per hidden neuron.

### Why Each Architecture is Larger

- **MLP**: One weight per connection. Minimal storage.
- **LSTM**: 4 gates (input, forget, output, cell) multiply connection weights by 4x, plus recurrent state and cell memory.
- **GRU**: 3 gates (update, reset, candidate) multiply connection weights by 3x, plus recurrent state. ~20% smaller than LSTM.
- **KAN**: Each edge stores B-spline coefficients (`GridSize + SplineDegree` = 6 per edge with G=5, k=1), plus a base weight and spline weight. Training adds gradient, delta weight, and previous delta weight for every learnable parameter.

All architectures remain well under 1.2 KB in Q8.8 fixed-point even in trainable form, making them suitable for embedded deployment.
