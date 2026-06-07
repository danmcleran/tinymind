---
title: Predictive Maintenance (AI4I 2020)
parent: Examples
nav_order: 50
layout: default
---

# Predictive Maintenance (AI4I 2020)

A binary machine-failure classifier trained on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), predicting whether a milling-machine reading represents an imminent failure.

## How it works

- Q16.16 fixed-point MLP, 7&nbsp;&rarr;&nbsp;8&nbsp;&rarr;&nbsp;1, ReLU hidden layer and a single sigmoid output (threshold 0.5 for the failure decision).
- Demonstrates the TinyMind `NeuralNet<>` feed-forward MLP trained and run end to end in `QValue` fixed-point — the train-and-deploy-on-an-MCU path applied to imbalanced binary classification.
- The 5 process features (air temp, process temp, rpm, torque, tool wear) are z-score normalized then scaled by 1/3 to sit in Q16.16's stable range, plus a 2-dim one-hot for the product variant. Because real failures are only ~3.4% of the data, training uses 50/50 balanced sampling so the net cannot collapse to the trivial majority classifier.

## Build and run

```bash
cd examples/predictive_maintenance
make release
make run
make plot      # needs matplotlib in an isolated env (venv/pyenv)
```

If `ai4i2020.csv` is present in the run directory (`make run` `cd`s into `./output/`) it is loaded directly; otherwise the program synthesizes 10,000 rows following the documented AI4I 2020 generative and failure-labelling rules (HDF, PWF, OSF, TWF, RNF), so the example trains end to end with no download. To use the real data, download `ai4i2020.csv` from the UCI page above and `cp` it into `./output/` before `make run`.

## Output

![Predictive maintenance training loss and test confusion matrix]({{ site.baseurl }}/assets/plots/predictive_maintenance.png)

The loss curve settles around 0.08 average error. The confusion matrix shows the balanced-sampling payoff: of 317 true failures, 282 are caught (recall ~0.89) at an overall test accuracy of ~0.87 and an F1 of ~0.68 — the model favors catching failures over avoiding false alarms, the right trade-off for maintenance.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/predictive_maintenance)
