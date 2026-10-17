---
title: Decision Trees & GBDT
layout: default
parent: Architectures
nav_order: 12
---

# Decision Trees & GBDT

> **Real-world use:** a pump controller decides on/off from a handful of temperature, pressure, and flow readings. A gradient-boosted tree ensemble walks root-to-leaf with nothing but integer compares and branches — no matmul, no activation tables — making it the cheapest model the microcontroller can run, and a better fit for tabular sensor data than a dense net.

tinymind provides an int8 **decision-tree** family — single trees and
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
| `cpp/qtree.hpp` | `QStaticDecisionTree` / `QStaticGBDT` | the same models from a compile-time node pool, run by an evaluation policy |

## Quantized splits are lossless

A decision-tree split compares one feature against a threshold. tinymind
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

## Compile-time models

When the trees are fixed at build time, the node pool can be a `constexpr`
array inside a type, and `QStaticDecisionTree` / `QStaticGBDT` run it under an
evaluation policy:

| Policy | How the pool runs | Pick it when |
|--------|-------------------|--------------|
| `QTreeInlineEval` (default) | every node is expanded at compile time into a compare of `x[feature]` against an immediate threshold and a branch, the nested if/else a code generator would write; the pool is never read at run time, so it is not stored | speed, and small-to-medium models |
| `QTreeWalkEval` | `walkQTree` over the pool at run time: one shared loop, the pool as data | large ensembles, where code per node would outgrow the data |

The two give the same result for every input, so switching is a one-word change:

```cpp
struct Model
{
    static constexpr tinymind::QTreeNode<int8_t> nodes[] =
    {
        {  0, 12,  1,  2, 0 },          // f0 <= 12 ? node1 : node2
        { -1,  0, -1, -1, 3 },          // leaf: 3
        { -1,  0, -1, -1, 7 },          // leaf: 7
    };
};

using Tree = tinymind::QStaticDecisionTree<int8_t, Model, 4, tinymind::QTreeInlineEval>;
const int32_t leaf = Tree::predict(x);  // x: const int8_t[4]
```

`QStaticGBDT` takes the ensemble's `tree_root`, `tree_class` and `base_score`
from the same type, as `constexpr` arrays next to `nodes`.

The model is checked when it compiles, under either policy: a split feature
that is not below `NumFeatures`, a child or root index outside the pool, or a
class index outside `NumClasses` is a build error rather than an out-of-bounds
read, and the inline policy also refuses a cyclic tree. Both types build at
C++14; there, `QTreeWalkEval` needs the usual namespace-scope definition of
each static array (`constexpr tinymind::QTreeNode<int8_t> Model::nodes[];`),
which C++17 makes implicit.

Unsigned features help the inline policy on Cortex-M0+: Thumb-1 has an
immediate-offset `LDRB` but only a register-offset `LDRSB`, so `uint8_t` inputs
save an instruction per node over `int8_t`.

See the [`gbdt_tabular_int8`]({{ site.baseurl }}/examples/gbdt_tabular_int8)
example for a 3-class ensemble, the int8-vs-float region agreement check, and
the decision-region map.
