---
title: Neuro-Fuzzy (ANFIS)
layout: default
parent: Architectures
nav_order: 13
---

# Neuro-Fuzzy Inference (ANFIS)

> **Real-world use:** a process controller has to justify itself. When the loop trips a safety review, "the network said so" is not an answer — but "rule 7 fired at 0.62, and rule 7 says *if inlet temperature is high and flow rate is low, reduce the setpoint*" is. TinyMind's ANFIS gives a learned model whose rules you can read back after every inference, in a few hundred bytes.

TinyMind provides **ANFIS** — the Adaptive Neuro-Fuzzy Inference System of
Jang (1993) — as a standalone composable layer. It is a Takagi-Sugeno fuzzy
system whose membership functions and rule consequents are learned from
data, so it sits between a rule base you hand-write and a dense net you
cannot interrogate.

| Header | Type | Role |
|--------|------|------|
| `cpp/anfis.hpp` | `Anfis` | the five-stage Takagi-Sugeno forward pass |
| `cpp/anfis.hpp` | `TriangularMembershipFunction` | `{a, b, c}` — arithmetic only |
| `cpp/anfis.hpp` | `TrapezoidalMembershipFunction` | `{a, b, c, d}` — arithmetic only |
| `cpp/anfis.hpp` | `GeneralizedBellMembershipFunction` | `{a, c}`, compile-time exponent — arithmetic only |
| `cpp/anfis.hpp` | `GaussianMembershipFunction` | `{c, sigma, 1/sigma}` — integer exp lookup table |
| `cpp/anfis.hpp` | `AnfisFullGridRuleTable` | fills the full `M^N` rule grid |
| `apps/anfis_train` | `anfis_train.py` | host-side hybrid trainer + header emitter |

## The five stages

```
1. fuzzify    mu[i][m]  = MembershipFunction(x[i])
2. fire       w[r]      = product over i of mu[i][ruleTable[r][i]]
3. normalize  wbar[r]   = w[r] / sum(w)
4. consequent f[r]      = p[r] . x + q[r]     (first order)
                        = c[r]                (zeroth order)
5. output     y         = sum over r of wbar[r] * f[r]
```

No state carries between calls, nothing is allocated, and every rule is
evaluated every time — so latency does not depend on the input data. That
matters more on a control loop than raw speed does.

## One divide, not one per rule

Stages 3 and 5 are **fused**:

```
y = sum(w[r] * f[r]) / sum(w)
```

A literal implementation would compute a reciprocal per rule. Fusing costs
one divide per output instead, and in a narrow Q format it is also far more
accurate — a per-rule `wbar` in Q8.8 quantizes each weight to 1/256 before
it ever multiplies a consequent. Q16.16 is the recommended deployment type.

If nothing fires — every input fell outside the support of every membership
function it is tested against — the output is zeroed rather than divided by
zero. `getTotalFiringStrength()` tells the two cases apart.

## The rule table is explicit, and that is the point

The rule base is a `uint8_t` table indexed `[rule][input]`, not an implicit
grid. The full grid is `M^N` rules: 3 membership functions over 4 inputs is
81 rules, over 8 inputs it is 6561. That explosion is the classic knock
against ANFIS, and published fuzzy hardware overwhelmingly stays under ~25
rules.

An explicit table lets an offline pruning pass ship only the rules that
carry weight. On the bundled Mackey-Glass benchmark, an oversized 81-rule
grid against 500 training samples overfits catastrophically, and pruning by
mean firing strength is what recovers it:

| rules kept | train RMSE | test RMSE |
|---|---|---|
| 81/81 | 0.000091 | 0.635650 |
| 55/81 | 0.000719 | 0.010929 |
| 35/81 | 0.002733 | 0.004798 |
| 21/81 | 0.004182 | 0.004793 |
| 13/81 | 0.007016 | 0.007482 |

Keeping 26% of the rules cuts test error by a factor of ~130.

Read that table qualitatively rather than to the digit: the unpruned 81-rule design matrix is rank deficient (rank 402 of 405, condition number 9.0e12), so which least-squares solution the premise trajectory lands on shifts with floating-point summation order. The shape is robust — orders of magnitude of overfit, recovered by pruning — while the exact unpruned figure is not. The shipped 16-rule model is a different regime: rank 80 of 80, condition number 1.9e5, reproducible exactly.

A `DontCareIndex` antecedent drops an input from a rule entirely,
contributing a grade of 1 — which is what a pruned antecedent looks like.

### Or skip the grid entirely

Because the rule table is explicit, it does not have to *be* a grid. The host
trainer's `build_scatter_anfis` runs Chiu subtractive clustering over the
training data and gives each cluster one rule, with one membership function per
input centered on that cluster. The table becomes the diagonal — rule `r` reads
membership function `r` of every input — and `cpp/anfis.hpp` runs it unchanged.

The rule count then follows the data's structure rather than the input count:

| inputs | grid at 3 MFs | subtractive clustering |
|---|---|---|
| 2 | 9 | 3 |
| 4 | 81 | 8 |
| 6 | 729 | 8 |
| 8 | 6561 | 14 |

On the bundled benchmark, 5 clustered rules and 65 parameters land within 18%
of the 16-rule grid's test error.

## Membership functions

Triangular, trapezoidal, and generalized-bell shapes use only compare, add,
multiply, and divide, so they hold at `TINYMIND_ENABLE_FLOAT=0` /
`TINYMIND_ENABLE_STD=0`. Pinning Jang's bell exponent `b` at compile time
replaces `pow()` with `BellExponent - 1` multiplies, which is what keeps the
bell in that group.

The Gaussian rides the integer exp lookup table (the same one
`SoftmaxActivationPolicy` uses), so it needs no FPU either — just the
matching `TINYMIND_USE_EXP_<F>_<Fr>` build macro. Float and double
specializations sit behind `FLOAT && STD`.

Overflow in a narrow Q format is headed off **structurally** rather than by
saturation: the bell clamps once `t >= 64`, and the Gaussian tests
`|x - c| >= 4*sigma` *before* the multiply — which is why `sigma` is carried
alongside its own reciprocal. Both types apply the identical 4-sigma cut, so
a model behaves the same in float and in fixed point.

## Reading the rules back

Firing strengths survive the call. `getFiringStrength(r)`,
`getNormalizedFiringStrength(r)`, `getDominantRule()`, and
`getMembershipGrade(input, mf)` report what the model just did, so an
application can log which rule owned an output, or refuse to act when no
rule fires strongly.

On the Mackey-Glass model, reading the rule base back shows rule 0 carrying
18% of the output and the eight odd-numbered rules together carrying under
10% — the model barely uses the upper half of its most recent input. A dense
net of the same size cannot make that statement about itself.

## The int8 tier costs memory rather than saving it

`cpp/qanfis.hpp` is the integer counterpart, and it is worth being blunt about
what it buys. Membership functions become 256-entry lookup tables, so their
shape stops mattering at runtime — but a table costs 512 bytes regardless of
how few parameters the shape it replaced had. On the bundled benchmark the
tables are 4096 bytes, 92.5% of the int8 model, and the whole thing is **9.9x
larger** than the same model in Q16.16 (4428 bytes against 448).

Per rule int8 is cheaper, 20 bytes against 24, so the crossover is about 1024
rules — three orders of magnitude past what this model or most published fuzzy
hardware uses.

Reach for the int8 tier when the rest of the graph is already int8 and you want
ANFIS to consume that directly, with no bridge and no float on the hot path.
Reach for Q16.16 when ANFIS is the model. See the
[int8 exemplar]({{ site.baseurl }}/examples/anfis_mackey_glass_int8) for the
measurements.

## Training is host-side

`cpp/anfis.hpp` is inference only. Jang's hybrid scheme solves the
consequents by **least squares** — they enter the output linearly given
fixed premises — and only then descends on the premises. A pseudo-inverse
over a `samples x rules*(inputs+1)` design matrix is not a microcontroller
workload, so `apps/anfis_train` does it on the host and emits the three
frozen arrays as a header.

See the [`anfis_mackey_glass`]({{ site.baseurl }}/examples/anfis_mackey_glass)
example: Jang's benchmark in 16 rules and 448 bytes of Q16.16, with test RMSE
0.004066 in double and 0.004081 in Q16.16.
