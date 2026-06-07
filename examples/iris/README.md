# Iris Species Classifier

Three-way iris species classifier trained on the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
using a Q16.16 fixed-point MLP from TinyMind.

## Network

- 4 inputs — sepal length/width, petal length/width (cm), z-score normalized
  (training-set statistics) then scaled by 1/3 to sit inside Q16.16's stable range
- 1 hidden layer of 8 ReLU neurons
- 3 sigmoid outputs — a one-hot encoding of the species; predicted class is argmax
- 30k iterations of uniform random sampling from the 80% training split

## Build and run

```bash
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` cd's into `./output`, so the Makefile copies the bundled `iris.data`
there first. The full 150-row dataset ships with the example (~4 KB).

## Output

- `output/iris_loss.csv` — training loss curve (iteration, avg |error|)
- `output/iris_confusion.csv` — 3×3 test confusion matrix
- `output/iris_scatter.csv` — per-test-sample petal length/width + true/predicted class
- `output/iris_confusion_behavior.png` — `plot.py` renders loss + confusion + petal scatter

Expected (seed = 7): clean convergence to ~0.015 avg error and **100% test
accuracy** (30/30) — Iris is linearly separable enough that the fixed-point MLP
nails the held-out split.

## TinyMind capability shown

`NeuralNet<>` feed-forward MLP (`MultilayerPerceptron`) trained and run entirely
in `QValue` Q16.16 fixed-point — the smallest end-to-end "train + deploy on an
MCU" path in the library. See `examples/xor` for the same pattern on a toy task.
