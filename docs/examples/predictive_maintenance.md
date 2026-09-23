---
title: Predictive Maintenance (AI4I 2020)
parent: Examples
nav_order: 50
layout: default
---

# Predictive Maintenance (AI4I 2020)

A binary machine-failure classifier trained on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), predicting whether a milling-machine reading represents an imminent failure.

## How it works

- Q16.16 fixed-point MLP, 10&nbsp;&rarr;&nbsp;24&nbsp;&rarr;&nbsp;1, ReLU hidden layer and a single sigmoid output (threshold 0.5 for the failure decision).
- Demonstrates the TinyMind `NeuralNet<>` feed-forward MLP trained and run end to end in `QValue` fixed-point — the train-and-deploy-on-an-MCU path applied to imbalanced binary classification.
- Inputs are the 5 process features (air temp, process temp, rpm, torque, tool wear), **3 physics-derived product features** (power ≈ rpm·torque, overstrain ≈ toolwear·torque, temperature gap), and a 2-dim one-hot for the product variant; all numeric inputs are z-score normalized then scaled by 1/3.
- **Training target:** AI4I labels a row a failure if any of five modes fired. Three (heat dissipation, power, overstrain) are deterministic functions of the sensor readings. The other two are random by construction: tool-wear failure (TWF) is a tool replaced at a random time between 200 and 240 min, and RNF is a 0.1% random failure. The net is trained on the three predictable modes and scored against the full `Machine failure` label, so the random failures still count against recall.
- **Sampling:** failures are ~3.4% of the data. Training draws 20% of its samples from the failure pool. A 50/50 split pushes the decision boundary deeper into healthy space: on the real CSV it raises false alarms by about half (439 vs 283 summed over 10 seeds).
- **Why the product features:** the AI4I failure modes are *products* of inputs (mechanical power drives PWF, tool-wear × torque drives OSF). A small ReLU MLP cannot synthesize a multiplication from raw features alone. Handing it those terms lifts precision from 0.59 to 0.66 on the real CSV (10-seed means).

## Build and run

```bash
cd examples/predictive_maintenance
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

If `ai4i2020.csv` is present in the run directory (`make run` `cd`s into `./output/`) it is loaded directly. Otherwise the program synthesizes 10,000 rows following the documented AI4I 2020 generative and failure-labelling rules (HDF, PWF, OSF, TWF, RNF), with distributions fitted to the real CSV (base rate, variant mix, rpm/torque correlation, tool-replacement window), so the example trains end to end with no download. To use the real data, download `ai4i2020.csv` from the UCI page above and `cp` it into `./output/` before `make run`.

## Output

![Predictive maintenance training loss and test confusion matrix]({{ site.baseurl }}/assets/plots/predictive_maintenance.png)

The plot is the default synthetic run (seed 7): 66 of 75 test failures caught with 23 false alarms out of 1925 healthy readings, for precision 0.74, recall 0.88 and F1 0.80.

On the real AI4I CSV the test split holds only about 60 failures, so one seed is noisy. Over seeds 1 to 10:

| Data | Precision | Recall | F1 |
|---|---|---|---|
| Real CSV | 0.66 ± 0.06 | 0.80 ± 0.06 | 0.72 ± 0.04 |
| Synthetic | 0.72 ± 0.04 | 0.80 ± 0.06 | 0.75 ± 0.04 |

52 of the 339 real failures (15%) come only from the random TWF/RNF modes, which no sensor reading predicts, so recall has a hard ceiling near 0.85.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/predictive_maintenance)
