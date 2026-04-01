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
| Elman RNN (2→3→1) | 3 | 1,056 | 384 | +672 (+175%) |
| LSTM (2→3→1) | 3 | 3,024 | 960 | +2,064 (+215%) |
| GRU (2→3→1) | 3 | 2,400 | 792 | +1,608 (+203%) |
| KAN (2→5→1, G=5, k=1) | 5 | 4,208 | 1,256 | +2,952 (+235%) |

### Using Q8.8 Fixed-Point (XOR Configuration)

Instance sizes in bytes for XOR network configurations using `QValue<8,8,true>` (signed Q8.8 fixed-point, 2 bytes per value).

| Architecture | Hidden Neurons | Trainable | Non-trainable | Training Overhead |
|---|---|---|---|---|
| MLP (2→3→1) | 3 | 328 | 144 | +184 (+128%) |
| Elman RNN (2→3→1) | 3 | 472 | 192 | +280 (+146%) |
| LSTM (2→3→1) | 3 | 952 | 384 | +568 (+148%) |
| GRU (2→3→1) | 3 | 808 | 336 | +472 (+140%) |
| KAN (2→5→1, G=5, k=1) | 5 | 1,192 | 416 | +776 (+187%) |

### Relative Size (vs MLP)

| | Trainable | Non-trainable |
|---|---|---|
| **double** | | |
| Elman / MLP | 1.0x | 1.1x |
| LSTM / MLP | 3.0x | 2.7x |
| GRU / MLP | 2.4x | 2.2x |
| KAN / MLP | 4.2x | 3.5x |
| **Q8.8** | | |
| Elman / MLP | 1.4x | 1.3x |
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
- **Elman RNN**: Same connection weights as MLP, plus a recurrent layer that stores previous hidden outputs and feeds them back as additional inputs.
- **LSTM**: 4 gates (input, forget, output, cell) multiply connection weights by 4x, plus recurrent state and cell memory.
- **GRU**: 3 gates (update, reset, candidate) multiply connection weights by 3x, plus recurrent state. ~20% smaller than LSTM.
- **KAN**: Each edge stores B-spline coefficients (`GridSize + SplineDegree` = 6 per edge with G=5, k=1), plus a base weight and spline weight. Training adds gradient, delta weight, and previous delta weight for every learnable parameter.

All architectures remain well under 1.2 KB in Q8.8 fixed-point even in trainable form, making them suitable for embedded deployment.

## Signal Processing Pipeline Sizes

Instance sizes in bytes for Conv1D, Pool1D, and Dropout layers. These are standalone composable layers that sit outside the neural network template.

### Conv1D

| Configuration | `double` | Q8.8 |
|---|---|---|
| Conv1D (100, kernel=5, stride=2, 8 filters) | 768 | 192 |
| Conv1D (100, kernel=5, stride=1, 4 filters) | 384 | 96 |

Conv1D stores kernel weights + biases and their gradients. Size = `2 * NumFilters * (KernelSize + 1) * sizeof(ValueType)`.

### MaxPool1D

| Configuration | Size (bytes) |
|---|---|
| MaxPool1D (96 input, pool=2, stride=2, 4 channels) | 1,536 |
| MaxPool1D (48 input, pool=2, stride=2, 1 channel) | 192 |
| MaxPool1D (6 input, pool=2, stride=2, 1 channel) | 24 |

MaxPool1D stores argmax indices (`size_t` per output) for backpropagation gradient routing. Size is independent of value type: `OutputSize * sizeof(size_t)`.

### AvgPool1D

AvgPool1D is stateless (1 byte). It has no per-instance storage since average gradients are computed directly from the pool size.

### Dropout

| Configuration | Size (bytes) |
|---|---|
| Dropout (192 elements, 50%) | 193 |
| Dropout (32 elements, 50%) | 33 |
| Dropout (5 elements, 50%) | 6 |

Dropout stores a boolean mask (1 byte per element) plus a training mode flag. Size is independent of value type: `Size + 1` bytes.

### Full Pipeline: Conv1D -> MaxPool1D -> Dropout

| Value Type | Conv1D (100, k=5, s=1, 4 filters) | MaxPool1D (96, p=2, s=2, 4 ch) | Dropout (192, 50%) | Total |
|---|---|---|---|---|
| `double` | 384 | 1,536 | 193 | **2,113** |
| Q8.8 | 96 | 1,536 | 193 | **1,825** |

The MaxPool1D dominates pipeline memory due to argmax index storage. For memory-constrained deployments, AvgPool1D eliminates this overhead entirely (1 byte vs 1,536 bytes for the same configuration).

## Binary and Ternary Dense Layer Sizes

Binary and ternary layers store weights as packed bits (1-bit or 2-bit per weight) for extreme memory reduction. During training, real-valued latent weights and gradients are also maintained. After training, only the packed weights and biases are needed for inference.

### BinaryDense (Trainable)

Trainable instances store packed binary weights, latent weights, latent gradients, biases, and bias gradients.

| Configuration | `double` | Q8.8 |
|---|---|---|
| BinaryDense (4, 2) | 168 | 44 |
| BinaryDense (16, 8) | 2,192 | 560 |
| BinaryDense (64, 16) | 16,768 | 4,288 |
| BinaryDense (32, 32) | 17,024 | 4,352 |

### TernaryDense (Trainable)

Trainable instances store packed ternary weights (2-bit), latent weights, latent gradients, biases, and bias gradients.

| Configuration | `double` | Q8.8 |
|---|---|---|
| TernaryDense (4, 2, 50%) | 168 | 44 |
| TernaryDense (16, 8, 50%) | 2,208 | 576 |
| TernaryDense (64, 16, 50%) | 16,896 | 4,416 |
| TernaryDense (32, 32, 50%) | 17,152 | 4,480 |

### Inference-Only Packed Weight Storage

After training, only the packed weights and biases are needed. This represents the minimum deployment footprint.

| Configuration | Binary (packed) | Ternary (packed) | Full-precision `double` | Full-precision Q8.8 |
|---|---|---|---|---|
| 64x16 weights + biases | 256 | 384 | 8,320 | 2,080 |
| 32x32 weights + biases | 384 | 512 | 8,448 | 2,112 |

### Weight Storage Compression Ratios (64x16 layer)

| Storage | Bytes | Compression vs `double` | Compression vs Q8.8 |
|---|---|---|---|
| Full `double` | 8,192 | 1x | — |
| Full Q8.8 | 2,048 | 4x | 1x |
| Packed binary (1-bit) | 128 | **64x** | **16x** |
| Packed ternary (2-bit) | 256 | **32x** | **8x** |

Binary packing achieves 64x compression over `double` and 16x over Q8.8. Ternary packing achieves 32x over `double` and 8x over Q8.8 while supporting weight sparsity (zero weights are skipped entirely in the forward pass).
