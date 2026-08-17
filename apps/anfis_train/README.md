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
| 0.000 | 81/81 | 0.000116 | 1.768326 |
| 0.001 | 58/81 | 0.000836 | 0.020733 |
| 0.005 | 39/81 | 0.002413 | 0.006387 |
| 0.010 | 25/81 | 0.004271 | 0.005330 |
| 0.020 | 13/81 | 0.007238 | 0.006822 |

Pruning to 25 rules cuts the test error by a factor of ~330 while shipping
31% of the rule base.

## Not implemented

* **Subtractive clustering / scatter partition.** Only grid partition
  (`grid_partition`, `full_grid_rules`) is provided. A scatter partition
  would fit the same `Anfis` container — one rule per cluster, with each
  cluster owning its own membership function per input — but it is not
  written.
* **Multiple outputs.** The parameter layout carries the output axis so the
  emitted header matches the C++ indexing, but only `n_outputs == 1` is
  trained.
* **int8.** This produces float/Q-format parameters. A quantized ANFIS
  (`qanfis.hpp`) is a separate piece of work.
