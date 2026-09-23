# Predictive Maintenance (AI4I 2020)

Binary machine-failure classifier trained on the [AI4I 2020 Predictive
Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
using a Q16.16 fixed-point MLP from TinyMind.

## Network

- 10 inputs — 5 process features (air temp, process temp, rpm, torque, tool
  wear), 3 physics-derived product features (power ≈ rpm·torque, overstrain ≈
  toolwear·torque, temperature gap), and a 2-dim one-hot for product variant
  (L, M; H = `[0, 0]`); all numeric inputs z-score normalized + 1/3 scaled
- 1 hidden layer of 24 ReLU neurons
- 1 sigmoid output (threshold 0.5 for failure prediction)
- 300k training iterations, 20% of them drawn from the failure pool

**Training target.** AI4I labels a row a failure if any of five modes fired.
Three of them (heat dissipation HDF, power PWF, overstrain OSF) are
deterministic functions of the sensor readings. The other two are not: TWF is
a tool replaced at a random wear time between 200 and 240 min, and RNF is a
0.1% random failure. Training on those as positives teaches the net to alarm
on high-wear samples, so the net learns the three predictable modes and is
scored against the full `Machine failure` label. The TWF and RNF rows it cannot
see still count against recall.

**Sampling.** Failures are ~3.4% of the data. The net sees 20% positives during
training: enough to learn the failure regions, without the decision boundary a
50/50 split pushes deep into healthy space. On the real CSV, 50/50 sampling
with every failure mode as the target gave precision 0.29 and F1 0.43. The
current setup gives precision 0.66 and F1 0.72 (10-seed means).

**Product features.** The AI4I failure modes are *products* of inputs
(mechanical power for PWF, tool-wear × torque for OSF). A small ReLU MLP cannot
synthesize a product from raw features, so feeding the products in directly
lifts precision from 0.59 to 0.66 and F1 from 0.67 to 0.72 on the real CSV
(10-seed means).

## Build and run

```bash
make release
make run
```

## Dataset

If `ai4i2020.csv` is present in the run directory (the example looks in
`./output/` since `make run` `cd`s there), it is loaded directly. Otherwise the
program synthesizes 10,000 rows following the documented AI4I 2020 generative
and failure-labelling rules (HDF, PWF, OSF, TWF, RNF) so the example can train
end-to-end without a download.

To use the real CSV:

```bash
# Download ai4i2020.csv from the UCI page above, then:
cp ai4i2020.csv ./output/
make run
```

Expected output (seed = 7):

```
synthetic: Train: 8000 (pos=258, neg=7742, by training target)  Test: 2000
           accuracy 0.984   precision 0.742   recall 0.880   F1 0.805
real CSV:  Train: 8000 (pos=239, neg=7761, by training target)  Test: 2000
           accuracy 0.978   precision 0.606   recall 0.717   F1 0.657
```

The test split holds only ~60 real failures, so a single seed is noisy. Over
seeds 1 to 10 the real CSV gives precision 0.66 ± 0.06, recall 0.80 ± 0.06 and
F1 0.72 ± 0.04. The synthetic set matches the real CSV's base rate, variant
mix, rpm/torque correlation and tool-replacement window, and gives F1
0.75 ± 0.04.
