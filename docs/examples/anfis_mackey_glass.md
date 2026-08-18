---
title: Mackey-Glass ANFIS
parent: Examples
nav_order: 67
layout: default
---

# Mackey-Glass ANFIS

Jang's original 1993 benchmark for ANFIS: predict `x(t+6)` of the Mackey-Glass chaotic time series from four delayed samples — `x(t-18)`, `x(t-12)`, `x(t-6)`, `x(t)`. Two generalized-bell membership functions per input over the full grid is `2^4 = 16` rules with first-order consequents: **448 bytes** of model in Q16.16. Built on `cpp/anfis.hpp` and `apps/anfis_train`.

## How it works

- **Training is host-side, inference is not.** Jang's hybrid scheme solves the consequents by least squares (they enter the output linearly given fixed premises) and only then descends on the premises. `apps/anfis_train/anfis_train.py` runs that loop in numpy and emits the frozen premise, rule-table, and consequent arrays as `anfis_model.hpp`; `cpp/anfis.hpp` never trains.
- **Two value types, one frozen model.** The example runs the same parameters in `double` and in Q16.16, so the fixed-point cost is measured rather than assumed. The C++ `double` result reproduces the Python trainer's number exactly, which cross-checks the two forward passes against each other.
- **The rules are readable.** Per-rule mean normalized firing strength over the held-out set says what fraction of the output each rule owned — the reason to reach for ANFIS over a dense net of the same size.
- **Arithmetic only.** Generalized bells with the exponent pinned at 1 need no transcendental function and no lookup table, so the shipped model would deploy unchanged at `FLOAT=0` / `STD=0`. The example enables `FLOAT` and `STD` only for its `double` reference and `<cmath>` error metrics.

## Results

| | value |
|---|---|
| rules | 16 |
| premise / consequent parameters | 16 / 80 |
| Q16.16 model size | 448 bytes |
| held-out samples | 500 |
| test RMSE (double) | 0.004066 |
| test RMSE (Q16.16) | 0.004081 |
| max abs Q16.16 vs double | 0.001677 |

Test RMSE is 0.43% of the target range; the fixed-point path costs 0.4% more error than the double reference.

## Build and run

```bash
cd examples/anfis_mackey_glass
make release
make run                 # writes output/*.csv
make plot                # needs matplotlib; a venv/pyenv works
make regenerate-model    # optional: refit with train.py (needs numpy)
make golden              # deterministic Q16.16 stream for the integration gate
```

The model is committed, so the build needs no Python.

`make golden` emits the byte stream that `unit_test/integration` locks byte-for-byte, the same gate the int8 exemplars carry. It contains only raw Q16.16 integers and rule indices — never formatted doubles, since the fixed-point path is reproducible across compilers and optimization levels while printed floating point is not (verified identical under `-O0`, `-g`, `-O3`, and `-O2 -ffast-math`). The rule indices ride along because the defuzzified output alone could mask a regression in the premise or t-norm stages.

## Output

![Mackey-Glass ANFIS]({{ site.baseurl }}/assets/plots/anfis_mackey_glass.png)

**Held-out fit.** The prediction tracks every swing of the chaotic series; the Q16.16 output is drawn as sparse markers because a third line would hide under the other two.

**Hybrid training.** Epoch 0 is the least-squares-only baseline (test RMSE 0.004579); tuning the premises alongside it reaches 0.004066. The late wobble is the fixed-size normalized premise step bouncing near the optimum.

**Rule base read back.** Rule 0 carries 18% of the output; the eight odd-numbered rules together carry under 10%. An odd rule index means "the second membership function of `x(t)`", so the model barely uses the upper half of its most recent input.

**Pruning an oversized grid.** Three membership functions per input is 81 rules and 405 consequent parameters against 500 training samples — it overfits catastrophically (train RMSE 0.000091, test RMSE **0.636**). Pruning by mean firing strength recovers it: 21 of 81 rules gives test RMSE 0.004793, a ~130x improvement while shipping 26% of the rule base. This is why `cpp/anfis.hpp` carries an explicit rule table rather than an implicit `M^N` grid. Read the figures qualitatively: the unpruned 81-rule design matrix is rank deficient (rank 402 of 405, condition number 9.0e12), so its exact test RMSE shifts with floating-point summation order, while the shipped 16-rule model is full rank (condition number 1.9e5) and reproducible exactly.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/anfis_mackey_glass)
