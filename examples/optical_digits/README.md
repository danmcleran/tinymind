# Optical Handwritten-Digit Classifier

Ten-way handwritten-digit classifier trained on the
[UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
using a Q16.16 fixed-point MLP from TinyMind.

Each sample is an 8×8 bitmap (64 pixels, intensity 0..16, flattened row-major)
produced by downsampling a 32×32 scan of a handwritten digit.

## Network

- 64 inputs — one per pixel, per-feature z-score normalized (training-set
  statistics) then scaled by 1/3 to sit inside Q16.16's stable range. Many
  border pixels are constant zero, so the normalizer guards `stdev < 1e-6`
  (mapping those features to a constant 0 input).
- 1 hidden layer of 32 ReLU neurons
- 10 sigmoid outputs — a one-hot encoding of the digit; predicted class is argmax
- 60k iterations of uniform random sampling from the training split

## Build and run

```bash
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

`make run` cd's into `./output`, so the Makefile copies the bundled
`optdigits.tra` (3823 train rows) and `optdigits.tes` (1797 test rows) there
first. Both files ship with the example.

## Output

- `output/digits_loss.csv` — training loss curve (iteration, avg |error|)
- `output/digits_confusion.csv` — 10×10 test confusion matrix
- `output/digits_samples.csv` — up to 40 sampled test images (raw 64 pixels +
  true/predicted label), a mix of correct and misclassified cases
- `output/digits_confusion_behavior.png` — `plot.py` renders the training loss
  curve next to the confusion-matrix heatmap (titled with overall test accuracy)
- `output/digits_grid.png` — `plot.py` renders the sampled 8×8 digit images in a
  5×8 grid, each titled `t=<true> p=<pred>` (red when misclassified)

Expected (seed = 7): clean convergence to ~0.001 avg error and **~96% test
accuracy** (1729/1797) on the held-out test split.

## TinyMind capability shown

`NeuralNet<>` feed-forward MLP (`MultilayerPerceptron`) trained and run entirely
in `QValue` Q16.16 fixed-point on a real 64-feature image classification task —
the same "train + deploy on an MCU" path as `examples/iris`, scaled up to a
larger input and a 10-class output. See `examples/iris` for the 4-input version
of the same pattern.
