---
title: Gas Sensor Array Drift
parent: Examples
nav_order: 55
layout: default
---

# Gas Sensor Array Drift

A six-way gas classifier that demonstrates **sensor drift** — the slow accuracy decay a fixed model suffers as metal-oxide gas sensors age — and then corrects it on the device from a few labelled samples, inspired by the UCI [Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/224) (16 chemo-resistive sensors × 8 features, recorded over 36 months in 10 batches).

## How it works

- Q16.16 fixed-point MLP, 128&nbsp;&rarr;&nbsp;32&nbsp;&rarr;&nbsp;6, ReLU hidden layer and 6 sigmoid outputs (one-hot, predicted class is the argmax).
- The same `NeuralNet<>` fixed-point train-and-deploy path as [Iris](iris.html), scaled up to a 128-input classification task, used to make sensor drift visible as a **drift curve**: train once on batch 1, then evaluate every later batch.
- The 128 inputs are z-score normalized using **batch-1 statistics**, which are then applied unchanged to every batch — normalization deliberately does not compensate for drift, which is the point. Each later batch applies a per-sensor-feature multiplicative gain and additive offset growing with the batch index, a documented stand-in for sensor aging.

## Build and run

```bash
cd examples/gas_sensor_drift
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

The dataset is synthetic (deterministic, seed 7; 6 gas classes with distinct 128-dim mean response vectors, 150 samples/class/batch over 10 batches), so there is no file to copy — `make run` `cd`s into `./output/` and runs the binary directly.

## Output

![Gas sensor training loss, accuracy drift curve over 10 batches, and batch-1 confusion matrix]({{ site.baseurl }}/assets/plots/gas_sensor_drift.png)

Batch 1 trains to a perfectly diagonal in-distribution confusion matrix (batch-1 accuracy 1.00), but the drift curve slopes clearly downward as the batch index grows — accuracy falls from ~1.0 on the early batches to ~0.79 by batch 10. The gases are identical throughout; only the sensor response distribution has moved, which is exactly the failure mode the dataset was created to study.

## Adapting to drift on the device

The example then adapts the shipped model to each later batch from a few labelled samples per class drawn from that batch, scores it on the rest of the batch, and averages over 10 random draws of those samples. The trained MLP is split into a frozen backbone (the 128&nbsp;&rarr;&nbsp;32 ReLU layer) and a head (32&nbsp;&rarr;&nbsp;6), and seven methods are compared: no adaptation; z-score statistics recomputed from the labelled samples or from the whole batch (labels unused); a [`PrototypeHead`]({{ site.baseurl }}/training/on-device-fine-tuning) of class-mean embeddings; a [`DenseHead`]({{ site.baseurl }}/training/on-device-fine-tuning) trained from the shipped head (head-only fine-tuning); the backbone's 32 biases trained under a frozen head; and biases and head together.

![Accuracy after on-device adaptation, by batch and by number of labelled samples, and batch-1 accuracy afterwards]({{ site.baseurl }}/assets/plots/gas_sensor_drift_adapt.png)

Batch 10 accuracy at seed 7, mean of 10 draws:

| Method | 1 per class | 5 per class | 20 per class |
|---|---|---|---|
| No adaptation | 0.794 | 0.794 | 0.791 |
| Re-normalize (labelled samples) | 0.889 | 0.923 | 0.938 |
| Re-normalize (whole batch) | 0.937 | 0.936 | 0.937 |
| `PrototypeHead` | 0.987 | 0.999 | 1.000 |
| `DenseHead` (head only) | 0.953 | 0.995 | 0.997 |
| Backbone biases only | 0.848 | 0.967 | 0.990 |
| Biases + `DenseHead` | 0.961 | 0.995 | 0.998 |

Over data seeds 1 to 10 the order holds: unadapted 0.766&nbsp;&plusmn;&nbsp;0.097, `PrototypeHead` 0.985&nbsp;&plusmn;&nbsp;0.008 from one labelled sample per class, `DenseHead` 0.936&nbsp;&plusmn;&nbsp;0.025 from one and 0.988&nbsp;&plusmn;&nbsp;0.005 from five. Re-normalizing cannot undo this drift, because by batch 10 it has flipped the sign of some features' gain. The embedding still separates the gases after drift; they have only moved, so a new class mean or a retrained head follows them, and training the backbone's biases as well adds almost nothing. After adapting to batch 10 the `DenseHead` still classifies batch 1 perfectly, while the `PrototypeHead` has partly forgotten it and re-normalizing has broken it; for drift that does not matter, since the sensors never return to batch 1.

The data is synthetic and cleanly separable, so treat this as a demonstration on one drift model rather than a general ranking: published on-device-learning work on real data finds that updating more than the last layer gives higher accuracy.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/gas_sensor_drift)
