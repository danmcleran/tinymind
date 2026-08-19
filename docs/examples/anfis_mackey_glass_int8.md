---
title: int8 ANFIS (Mackey-Glass)
parent: Examples
nav_order: 68
layout: default
---

# int8 ANFIS (Mackey-Glass)

The same model as [Mackey-Glass ANFIS]({{ site.baseurl }}/examples/anfis_mackey_glass), run through `cpp/qanfis.hpp` — and a deliberate negative result.

**int8 is 9.9x LARGER than Q16.16 for this model.** That is the point of the exemplar. Everywhere else in TinyMind, moving to int8 shrinks a model; here it does not, and measuring that is more useful than letting the assumption ride.

![tiers]({{ site.baseurl }}/assets/plots/anfis_mackey_glass_int8.png)

## Results

| tier | test RMSE | model bytes |
|---|---|---|
| float reference | 0.003997 | — |
| Q16.16 | 0.003994 | **448** |
| int8 | 0.008584 | **4428 (9.9x)** |
| int8, no ridge | 0.025452 | — |

Accuracy holds — int8 costs about 2.1x the float reference's error on a target spanning 0.94. Memory does not.

## Why int8 is bigger

An int8 layer's parameters are usually its weights, and halving their width halves the model. ANFIS spends most of its parameters on membership functions instead, and `cpp/qanfis.hpp` evaluates those through a **256-entry lookup table per (input, membership function) pair** — which is exactly what makes the membership shape irrelevant at runtime.

Those tables cost 512 bytes each regardless of how few parameters the shape they replaced had. A Q16.16 generalized bell is two numbers, eight bytes; its int8 lookup table is 512. Here the tables are **4096 bytes, 92.5% of the int8 model**.

Per *rule* int8 is genuinely cheaper (20 bytes against 24), so the crossover is the fixed table cost divided by the per-rule saving: **int8 overtakes Q16.16 at about 1024 rules**. This model has 16, and published fuzzy hardware overwhelmingly ships under 25 — so in practice the Q-format tier is the smaller one.

## When the int8 tier is worth it

Not for footprint — for **pipeline consistency**. An int8 frontend (a quantized conv stack, an int8 feature extractor) can feed ANFIS directly with no `qbridge` conversion and no float on the hot path. If the rest of the graph is already int8, paying 4 KB to keep it integer end to end may be the right call. If ANFIS is the whole model, use Q16.16.

## The ridge lesson, measured

| consequent solve | largest coefficient | int8 test RMSE |
|---|---|---|
| ridge = 1e-6 | 9.58 | 0.008584 |
| ordinary least squares | 168.98 | 0.025452 |

Unregularized least squares produces large coefficients that nearly cancel. In float that is harmless — both fits score about 0.004. In int8 it is not: the input carries half a quantization step of error, and a coefficient of 169 amplifies it past the output quantum, breaking the cancellation the fit depends on. This is why `apps/anfis_train` carries a `ridge` parameter.

## Build and run

```bash
cd examples/anfis_mackey_glass_int8
make release
make run                 # writes output/*.csv
make golden              # deterministic stream for the integration gate
make plot                # needs matplotlib
make regenerate-model    # optional: refit with train.py (needs numpy)
```

`make run` fails if int8 error exceeds 4x the float reference, so the comparison is a gate rather than a printout. The golden stream carries the footprint numbers alongside the outputs, since the memory comparison is the headline.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/anfis_mackey_glass_int8)
