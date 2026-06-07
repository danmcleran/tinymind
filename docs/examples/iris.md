---
title: Iris Species Classifier
parent: Examples
nav_order: 51
layout: default
---

# Iris Species Classifier

A three-way iris species classifier trained on the classic [UCI Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) (sepal/petal measurements for setosa, versicolor, and virginica).

## How it works

- Q16.16 fixed-point MLP, 4&nbsp;&rarr;&nbsp;8&nbsp;&rarr;&nbsp;3, ReLU hidden layer and 3 sigmoid outputs (one-hot, predicted class is the argmax).
- The smallest end-to-end "train + deploy on an MCU" classification path in TinyMind — a `NeuralNet<>` feed-forward MLP trained and run entirely in `QValue` fixed-point.
- The 4 input features are z-score normalized using training-set statistics, then scaled by 1/3 to sit inside Q16.16's stable range; 30k iterations of uniform random sampling from the 80% training split.

## Build and run

```bash
cd examples/iris
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

The full 150-row dataset (~4&nbsp;KB) ships with the example as `iris.data`; the Makefile copies it into `./output/` before the run, so there is nothing to download.

## Output

![Iris training loss, test confusion matrix, and petal scatter]({{ site.baseurl }}/assets/plots/iris_classifier.png)

The loss converges cleanly to ~0.015 average error and the confusion matrix is perfectly diagonal — **100% test accuracy** (30/30). The petal length-vs-width scatter shows why: the three species form well-separated clusters that the fixed-point MLP nails on the held-out split.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/iris)
