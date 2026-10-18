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
| scikit-learn | `DecisionTree{Classifier,Regressor}`, `RandomForest{Classifier,Regressor}`, `ExtraTrees{Classifier,Regressor}`, `GradientBoosting{Classifier,Regressor}` |
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
  `FeatureQuant.from_data(X, dtype)` calibrates each feature's min/max.
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
