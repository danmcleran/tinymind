# tinymind_import_trees (scikit-learn / XGBoost / LightGBM -> tinymind trees)

Converts a trained tree model into a C++ header of compile-time node pools for
`tinymind::QStaticDecisionTree` / `QStaticGBDT` ([`cpp/qtree.hpp`](../../cpp/qtree.hpp)),
runnable under any of the qtree evaluation policies:

| Policy | Trade-off |
|---|---|
| `QTreeInlineEval` (default) | fastest: every node compiles to a compare against an immediate and a branch |
| `QTreeCompactEval` | smallest: a packed copy, 4-byte decision nodes and a deduplicated leaf table |
| `QTreeWalkEval` | the 12-byte `QTreeNode` pool walked at run time |

The header builds at C++14, with no float support unless you use its input
quantization helper.

## Supported models

| Library | Models |
|---|---|
| scikit-learn | `DecisionTree{Classifier,Regressor}`, `RandomForest{Classifier,Regressor}`, `ExtraTrees{Classifier,Regressor}`, `GradientBoosting{Classifier,Regressor}`, `IsolationForest` |
| XGBoost | `XGBClassifier`, `XGBRegressor`, `Booster` (gbtree) |
| LightGBM | `LGBMClassifier`, `LGBMRegressor`, `Booster` |

Not supported: missing values (NaN), categorical splits, multi-output trees,
XGBoost dart / gblinear, LightGBM linear trees and rf mode. The importer raises
a `ValueError` for each rather than exporting something subtly different.

## Use

```python
import sys; sys.path.insert(0, "apps/import_trees")
from tinymind_import_trees import FeatureQuant, export_trees

clf.fit(X_train, y_train)
model = export_trees(clf, X_train, "model.hpp", namespace="my_model",
                     quant=FeatureQuant.from_data(X_train, "int8"))
print(model.report)        # trees, nodes, agreement with the original
model.verify(X_test)       # compile with g++ and check every policy is bit-exact
print(model.agreement(X_test))
```

On the device:

```cpp
#include "model.hpp"

my_model::InputType q[my_model::NumFeatures];
my_model::quantizeFeatures(x, q);                    // float -> grid (needs TINYMIND_ENABLE_FLOAT)
std::size_t cls = my_model::predict(q);              // default: QTreeInlineEval
std::size_t small = my_model::predict<tinymind::QTreeCompactEval>(q);
```

What the header provides depends on the model:

| Model | `forward(q, out)` | `predict(q)` |
|---|---|---|
| Classifier forest / tree (hard vote) | votes per class | most votes, ties to the lowest class |
| Boosted classifier, soft-vote forest | int32 raw score per class | largest score (multiclass) or score > 0 (binary) |
| Regressor | int32 raw value; `raw / kOutputScale` is the prediction | - |

`kClassLabels` maps a predicted index back to the original label when the
labels are integers; `kOutputScale` (behind `TINYMIND_ENABLE_FLOAT`) converts
raw scores back to the model's units.

## How the integer model is built

* **Features** go onto a per-feature integer grid, `q = round(x / scale) + zero_point`
  (ties away from zero, clamped), in `int8`, `uint8` or `int16`.
  `FeatureQuant.integer(n)` is for features that already are integers (ADC
  counts, pixels, bins): scale 1, and every split is reproduced exactly.
  `FeatureQuant.from_data(X, dtype)` calibrates each feature's min/max onto
  the whole grid. A range that excludes 0 needs a zero point outside
  `[qmin, qmax]`, and it gets one (it is `int32` in the header). Only a
  feature so far from 0 that `quantizeFeatures`' int32 sum could overflow
  (`|zero_point| > 2^31 - 1 - 10^9`) is widened to a range that includes 0.
* **Thresholds.** A split `x <= t` (scikit-learn, LightGBM) becomes
  `q <= floor(t/scale + zp)`, and XGBoost's `x < t` becomes
  `q <= ceil(t/scale + zp) - 1`, which is exact on the grid. A rounding grid
  cell straddles the threshold, though, and with a calibrated grid the samples
  that land in that one cell can sit on both sides. That cell follows the
  majority of the calibration samples in it. This matters most for XGBoost,
  whose thresholds are data values, so many samples lie exactly on one. A
  split that sends the whole grid one way is removed.
* **Leaves.** Classifier forests vote by default (one class per tree, like
  emlearn's inline forests). scikit-learn's own `predict` averages class
  probabilities instead, which can differ on close calls; `voting="soft"`
  reproduces that by giving every tree one additive copy per class, which
  costs that many times the pool. Boosted models and regressors add int32
  leaves scaled by a power of two chosen so that no sum can overflow.
* **Base score.** It is not read from library-specific metadata, which differs
  between versions and objectives. It is measured as the model's raw output
  minus the summed tree outputs on the calibration data, and must come out the
  same for every sample. A tree that was parsed wrongly shows up here as a
  base that varies, and the export is refused.

Agreement with the original is exact on integer features up to int32 leaf
rounding. On calibrated float features, a sample within one grid step of a
threshold can flip; `model.agreement(X)` reports it on whatever data you give
it, and should be checked on held-out data. Compiled with `-ffast-math`,
`quantizeFeatures` may no longer match the importer's reference.

## IsolationForest (anomaly detection)

An `IsolationForest` exports as an additive ensemble whose sum is the forest's
total path length.

* **Leaves.** Each leaf holds its depth plus c(n). Here n is the number of
  training samples that reached the leaf, and c(n) is the average path length
  of an unbuilt subtree over n samples. That is the quantity
  `score_samples` adds up.
* **Feature subsets.** Each tree's feature subset (`estimators_features_`) is
  mapped back to the input's columns.
* **Base check.** The base check compares the summed path length with
  `-T c(max_samples) log2(-score_samples)`, so a parsing error is refused
  here, as it is for the other models.
* **Threshold.** The header adds `kAnomalyThreshold`, which is
  scikit-learn's `offset_` moved into path-length units:
  `score_samples < offset_` is the same test as
  `sum < -T c(max_samples) log2(-offset_)`.
* **`predict()`** returns 1 for an outlier (scikit-learn's `predict() == -1`)
  and 0 for an inlier.
* **`kAveragePathLength`** is c(max_samples). With it,
  `-2^(-(raw / kOutputScale) / (NumTrees * kAveragePathLength))` recovers
  `score_samples`. On the host, `model.score_samples(X)` does the same.

```python
iso = IsolationForest(n_estimators=25, max_samples=64, contamination=0.01).fit(X_normal)
model = export_trees(iso, X_normal, "anomaly.hpp", namespace="anomaly",
                     quant=FeatureQuant.from_data(X_normal, "int8"))
```

On integer features, `predict` matches scikit-learn's sample for sample and the
score matches to 1e-6. On a calibrated int8 grid, the tests hold agreement
above 99% and the score within 0.02 on held-out data.

Isolation trees grow to full depth, about 2 x `max_samples` nodes each, so
`n_estimators` and `max_samples` set the flash size. Measured for a Cortex-M0+
at `-Os` with 8 features, using `QTreeWalkEval` (12-byte nodes; with these
deep trees, `QTreeCompactEval` comes out larger):

| Trees | `max_samples` | Nodes | Flash |
|---|---|---|---|
| 25 | 64 | 1,257 | 15.8 KB |
| 50 | 64 | 2,606 | 32.7 KB |
| 100 | 256 (scikit-learn's defaults) | 13,720 | 167 KB |

On a 32 KB part, about 25 trees of 64 samples is the practical ceiling. For a
smaller detector there is `cpp/qanomaly.hpp`: a k-means detector that learns
on the device takes 1.3 KB of code for 32 features.

## Verification

`model.verify(X, cxx="g++", std="c++14")` generates a harness, compiles the
header with `-Wall -Wextra -Werror -Wpedantic`, runs every policy over `X`, and
asserts that the raw outputs, the predictions and `quantizeFeatures` are
bit-exact with the importer's own integer reference (`model.predict_raw`).

## Tests

```
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn xgboost lightgbm
make test PY=.venv/bin/python
```

31 tests:
* the threshold rules against brute force;
* exact agreement on integer features for every classifier family;
* agreement floors on held-out calibrated data;
* regressors;
* g++ and clang++ at C++14 and C++17;
* uint8 / int16 grids;
* a freestanding Cortex-M0+ cross-compile without float;
* the rejections;
* a planted tree-parsing defect that the base-score check must catch.

XGBoost and LightGBM tests are skipped when those packages are missing, and
compile checks when the compilers are. These tests are not run in CI.
