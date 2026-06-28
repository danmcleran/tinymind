# gbdt_tabular_int8

An int8 **gradient-boosted decision tree (GBDT)** ensemble on a 2-feature,
3-class tabular problem — the cheapest model family on a microcontroller. Each
prediction is a handful of integer compares and a branch: no MACs, no matmul,
no activation tables. Built on [`cpp/qtree.hpp`](../../cpp/qtree.hpp)
(`QDecisionTree`, `QGBDT`).

## How it works

- The ensemble is a small additive set of shallow trees (as exported from an
  XGBoost / LightGBM-style trainer). Each tree contributes its leaf value to
  its target class's logit; **argmax** over the per-class logits is the
  prediction. Single-tree / single-output regression is the degenerate case.
- Thresholds are quantized onto the int8 feature grid. Because an affine map is
  monotone, `x_q <= t_q` matches `x_real <= t_real`, so the int8 tree
  reproduces the float tree's split decisions **exactly** — the only place they
  can differ is a sample landing within one int8 grid step (~0.008) of a
  threshold.
- The driver sweeps a dense 2D grid in both float (real thresholds) and int8
  (quantized thresholds), reports how often they agree (~98% — the rest are
  boundary cells), and writes the int8 decision-region map to a CSV.

## Build & run

```bash
make            # debug build
make run        # int8 vs float GBDT agreement over the grid + probe predictions
make golden     # probe predictions for regression
make plot       # 2D decision-region map PNG
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host
calibration — the runtime tree path is pure integer and freestanding-safe).

## Output

![int8 GBDT decision regions](output/gbdt_tabular_int8.png)

The 2D map colors each feature-grid cell by the int8 GBDT's argmax class —
three bands (class 0 / 1 / 2) split on feature 0, with feature 1 nudging the
boundaries through the additive `f1` trees. That additive combination of
shallow trees is exactly what gradient boosting builds.

## Bring your own model

`QTreeNode` is a flat `{feature, threshold, left, right, leaf_value}` record and
trees share one node pool, which is the layout a trained ensemble exports to. To
deploy a real GBDT: dump its trees, quantize each threshold onto your feature
grid (`quantize<int8_t>(thr, scale, zero_point, ...)`), keep the leaf values as
int32 logits, and point `QGBDT` at the pool — inference is unchanged.
