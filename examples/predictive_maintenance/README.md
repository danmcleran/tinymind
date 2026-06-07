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
- 80k iterations of 50/50 balanced sampling (failures are ~3.4% of the
  real dataset, so uniform sampling learns the trivial majority classifier)

The AI4I failure modes are *products* of inputs (mechanical power for PWF,
tool-wear × torque for OSF). A small ReLU MLP cannot synthesize those products
from raw features, so feeding them in directly is what lifts precision from
~0.55 to ~0.80 (F1 0.68 → 0.84) while keeping recall ~0.89.

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

Expected output on the synthetic path (seed = 7):

```
Train: 8000 (pos=~1300, neg=~6700)  Test: 2000
iter  80000   avg|err| ~ 0.05
accuracy ~ 0.95   precision ~ 0.80   recall ~ 0.89   F1 ~ 0.84
```
