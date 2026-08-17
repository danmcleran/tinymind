# anfis_train (host ANFIS trainer -> TinyMind `anfis.hpp`)

Phase C tooling. `cpp/anfis.hpp` is inference only — it takes frozen premise
parameters, a rule table, and consequent parameters. This module produces
those three arrays and emits them as a C++ header.

numpy only. No torch, no scipy.

## Why training is host-side

Jang's hybrid scheme treats the two parameter sets differently, because they
are different:

* **Consequents** enter the output *linearly* given fixed premises, so they
  are solved exactly by least squares — one pseudo-inverse, no iteration and
  no learning rate.
* **Premises** do not, so they descend on the gradient of the same squared
  error.

Alternating the two converges far faster than descending on everything,
which is the point of the hybrid rule. It is also why this does not belong
on a microcontroller: a pseudo-inverse over a
`samples x rules*(inputs+1)` design matrix is not an MCU workload.

## Layout

```
apps/anfis_train/
    README.md
    anfis_train.py     # core module
```

## Quick example

```python
import numpy as np
from anfis_train import build_grid_anfis, emit_model_header, rmse

X = ...   # [samples, inputs]
y = ...   # [samples]

model = build_grid_anfis(X, n_mfs=2)          # grid partition, full rule grid
history = model.fit(X, y, epochs=200, step=0.005)
print("train RMSE", rmse(model.predict(X), y))

emit_model_header("anfis_model.hpp", model, namespace="my_model")
```

The emitted header carries `Premise`, `RuleTable`, and `Consequent` in
exactly the layout `tinymind::Anfis` indexes, plus the shape constants:

```cpp
#include "anfis_model.hpp"

typedef tinymind::GeneralizedBellMembershipFunction<double, 1> MfType;
typedef tinymind::Anfis<double,
                        my_model::NumberOfInputs,
                        my_model::NumberOfMembershipFunctionsPerInput,
                        my_model::NumberOfRules,
                        MfType, true, my_model::NumberOfOutputs> AnfisType;

AnfisType anfis(my_model::Premise, my_model::RuleTable, my_model::Consequent);
```

For a Q-format build, convert the arrays at the call site with
`tinymind::ValueConverter<double, QType>` — the header always emits
double, matching how the rest of TinyMind treats host-produced constants.

## Membership functions

Generalized bells with the exponent pinned at 1, matching
`GeneralizedBellMembershipFunction<ValueType, 1>`:

```
u     = (x - c) / a
mu(x) = 1 / (1 + u^2)
```

The bell needs no transcendental function, so a model trained here deploys
into the freestanding (`TINYMIND_ENABLE_FLOAT=0` / `TINYMIND_ENABLE_STD=0`)
corner unchanged. The `mu = 0` clamp at `u^2 >= 64` is applied host-side too,
so the trained model and the device agree on the tails.

## The premise step is normalized

`fit(step=...)` scales the `(dE/da, dE/dc)` gradient to unit L2 norm before
stepping, so `step` is the distance the premise parameters travel per epoch
in their own units — not a raw learning rate.

This matters more than it sounds. The least-squares step drives the residual
down first, which leaves the premise gradient several orders of magnitude
smaller than the parameters it acts on (on the bundled Mackey-Glass problem,
gradients around 1e-5 against parameters around 0.5). A plain learning rate
tuned to move anything on one problem does nothing on the next, or diverges.

## Rule pruning

`rule_importance(X)` scores each rule by its mean normalized firing strength
over a dataset; `prune(X, y, threshold)` drops the rules below the threshold
and re-solves the consequents over the survivors.

This is the reason `cpp/anfis.hpp` carries an explicit `uint8_t` rule table
instead of an implicit `M^N` grid. On the bundled benchmark, an oversized
3-membership-function grid over 4 inputs — 81 rules, 405 consequent
parameters, 500 training samples — overfits catastrophically, and pruning is
what recovers it:

| threshold | rules kept | train RMSE | test RMSE |
|---|---|---|---|
| 0.000 | 81/81 | 0.000091 | 0.635650 |
| 0.001 | 55/81 | 0.000719 | 0.010929 |
| 0.005 | 35/81 | 0.002733 | 0.004798 |
| 0.010 | 21/81 | 0.004182 | 0.004793 |
| 0.020 | 13/81 | 0.007016 | 0.007482 |

Pruning to 21 rules cuts the test error by a factor of ~130 while shipping
26% of the rule base.

**Read this table qualitatively, not to the digit.** The unpruned 81-rule
design matrix is rank deficient — rank 402 of 405 columns, condition number
9.0e12 — so which least-squares solution the 200-epoch premise trajectory
lands on is sensitive to floating-point summation order. Reordering one
`numpy` reduction moved the unpruned test RMSE from 1.77 to 0.64 while
leaving the shipped 16-rule model bit-stable. What is robust is the shape:
the full grid overfits by two to three orders of magnitude, and pruning
recovers it to roughly 0.005. The shipped model is a different regime
entirely — rank 80 of 80, condition number 1.9e5, and reproducible exactly.

## Membership function policies

Shape is a policy, chosen at construction:

```python
model = build_grid_anfis(X, n_mfs=2, mf=TriangularMembership)
```

| policy | parameters | `cpp/anfis.hpp` type |
|---|---|---|
| `BellMembership` (default) | `{a, c}` | `GeneralizedBellMembershipFunction<ValueType, 1>` |
| `TriangularMembership` | `{a, b, c}` | `TriangularMembershipFunction<ValueType>` |

Neither needs a transcendental function, so either deploys at `FLOAT=0` /
`STD=0`.

**Prefer bells for the descent.** A bell is smooth and non-zero everywhere
inside its clamp, so the premise gradient never vanishes just because a
membership function drifted away from the data. A triangle has compact
support: once it no longer covers any sample, both `mu` and `d(mu)/d(param)`
are exactly zero there and it can never come back. Triangles are also
non-differentiable at the peak and at each foot; those kinks are measure-zero
and reported as a zero derivative.

On the bundled benchmark (4 inputs, 2 membership functions, 16 rules):

| policy | test RMSE, least squares only | after hybrid training |
|---|---|---|
| bell | 0.004579 | **0.004066** |
| triangular | 0.008551 | **0.007702** |

Both improve under premise descent; the bell reaches roughly half the error.
Reach for triangles when the deployment target wants the compact support —
`cpp/anfis.hpp` bails out of the product t-norm on the first zero grade, so
compact support buys real cycles on a large rule base.

Adding a shape means adding a class with `NumberOfParameters`,
`parameter_names`, `cpp_type`, `evaluate`, `gradients`, `initialize`, and
`project`. Check any new `gradients` against central finite differences —
but jitter the knots off the data first, since `initialize` places the middle
membership functions' feet exactly on the observed min and max, and
differencing across a kink converges to the mean of the one-sided
derivatives rather than to either of them.

## Not implemented

* **Subtractive clustering / scatter partition.** Only grid partition
  (`initialize`, `full_grid_rules`) is provided. A scatter partition
  would fit the same `Anfis` container — one rule per cluster, with each
  cluster owning its own membership function per input — but it is not
  written.
* **Multiple outputs.** The parameter layout carries the output axis so the
  emitted header matches the C++ indexing, but only `n_outputs == 1` is
  trained.
* **int8.** This produces float/Q-format parameters. A quantized ANFIS
  (`qanfis.hpp`) is a separate piece of work.
