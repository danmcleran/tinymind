# Predictive Maintenance (AI4I 2020)

Binary machine-failure classifier trained on the [AI4I 2020 Predictive
Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
using a Q16.16 fixed-point MLP from TinyMind.

## Network

- 7 inputs — 5 process features (air temp, process temp, rpm, torque, tool wear)
  z-score normalized + 1/3 scaled, plus a 2-dim one-hot for product variant
  (L, M; H = `[0, 0]`)
- 1 hidden layer of 8 ReLU neurons
- 1 sigmoid output (threshold 0.5 for failure prediction)
- ~40k iterations of 50/50 balanced sampling (failures are ~3.4% of the
  real dataset, so uniform sampling learns the trivial majority classifier)

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
iter  40000   avg|err| ~ 0.08
accuracy ~ 0.87   recall ~ 0.89   F1 ~ 0.68
```
