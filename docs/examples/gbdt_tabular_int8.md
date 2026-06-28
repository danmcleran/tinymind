---
title: int8 GBDT (tabular)
parent: Examples
nav_order: 66
layout: default
---

# int8 GBDT (tabular)

An int8 gradient-boosted decision tree (GBDT) ensemble on a 2-feature, 3-class tabular problem — the cheapest model family on a microcontroller. Each prediction is a handful of integer compares and a branch: no MACs, no matmul, no activation tables. Built on `cpp/qtree.hpp` (`QDecisionTree`, `QGBDT`).

## How it works

- The ensemble is a small additive set of shallow trees (as a trainer like XGBoost / LightGBM would export). Each tree contributes its int32 leaf value to its target class's logit; **argmax** over the per-class logits is the prediction. A single tree / single output (`NumClasses = 1`) is the degenerate regression case.
- **Thresholds quantize losslessly in the comparison.** A split compares the feature against an int8 threshold on the same affine grid; because an affine map is monotone, `x_q <= t_q` matches `x_real <= t_real`, so the int8 tree reproduces the float tree's decisions exactly — the only place they can differ is a sample within one int8 grid step (~0.008) of a threshold.
- **MCU-perfect.** The runtime path is pure integer (compare + branch), freestanding-safe — no float, no LUT, no MAC. Nodes are a flat `{feature, threshold, left, right, leaf_value}` array and trees share one pool, the layout a trained ensemble exports to.
- The driver sweeps a dense 2D grid in float (real thresholds) and int8 (quantized thresholds) and reports ~98% agreement — the remaining couple percent are grid cells sitting on a split boundary.

## Build and run

```bash
cd examples/gbdt_tabular_int8
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host calibration). `make golden` writes the probe predictions to `output/gbdt_tabular_int8.golden`.

## Output

![int8 GBDT decision regions]({{ site.baseurl }}/assets/plots/gbdt_tabular_int8.png)

The 2D map colors each feature-grid cell by the int8 GBDT's argmax class — three bands (class 0 / 1 / 2) split on feature 0, with feature 1 nudging the boundaries through the additive `f1` trees. That additive combination of shallow trees is exactly what gradient boosting builds.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/gbdt_tabular_int8)
