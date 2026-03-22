# NeuralNetwork vs MultilayerPerceptron Size Comparison

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
