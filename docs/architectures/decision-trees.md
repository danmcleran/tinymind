---
title: Decision Trees & GBDT
layout: default
parent: Architectures
nav_order: 12
---

# Decision Trees & GBDT

TinyMind provides an int8 **decision-tree** family — single trees and
gradient-boosted ensembles — as a non-neural model family. Tree inference is
the cheapest model a microcontroller can run: a walk from root to leaf is a
handful of integer compares and a branch, with **no MACs, no matmul, and no
activation tables**. They are the right tool for tabular sensor data where a
dense net is overkill.

| Header | Type | Role |
|--------|------|------|
| `cpp/qtree.hpp` | `QTreeNode` | flat `{feature, threshold, left, right, leaf_value}` node record |
| `cpp/qtree.hpp` | `QDecisionTree` | one tree: walk to a leaf, return its int32 value |
| `cpp/qtree.hpp` | `QGBDT` | additive ensemble: per-class logit accumulation + argmax |

## Quantized splits are lossless

A decision-tree split compares one feature against a threshold. TinyMind
quantizes the threshold onto the **same int8 affine grid** as the input
feature, and an affine map is monotone:

```
x_q <= t_q   <=>   scale*(x_q - zp) <= scale*(t_q - zp)   <=>   x_real <= t_real
```

So quantizing the threshold preserves the split decision exactly — there is no
accuracy loss in the comparison itself. The int8 tree only ever disagrees with
the float tree on a sample whose feature lands within one grid step (~0.008 for
a `[-1, 1]` int8 grid) of a threshold. Leaf values are kept int32 so a boosted
ensemble can accumulate them into per-class logits without overflow.

## Single tree

`QDecisionTree` walks a flat node array from `nodes[0]`: at an internal node
(`feature >= 0`) it branches left if `x[feature] <= threshold`, else right; at a
leaf (`feature < 0`) it returns `leaf_value`. The walk is bounded by the node
count so a malformed (cyclic) tree terminates rather than spinning.

## Gradient-boosted ensemble

`QGBDT` is the additive ensemble. `NumTrees` trees share one node pool;
`tree_root[t]` is tree `t`'s root index and `tree_class[t]` the class whose
logit it contributes to. `forward()` seeds the per-class logits from an optional
`base_score` and adds every tree's leaf value into its class; `predict()`
returns the argmax. Single-output regression is `NumClasses == 1` (the logit is
the regression value). This is exactly the structure an XGBoost / LightGBM model
exports — dump the trees, quantize each threshold onto the feature grid, keep
the leaf values as int32 logits, and point `QGBDT` at the pool.

See the [`gbdt_tabular_int8`]({{ site.baseurl }}/examples/gbdt_tabular_int8)
example for a 3-class ensemble, the int8-vs-float region agreement check, and
the decision-region map.
