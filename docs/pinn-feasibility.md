---
title: PINNs (Physics-Informed Neural Networks)
layout: default
nav_order: 9
---

# TinyMind for Physics-Informed Neural Networks

This page assesses how TinyMind could be used to build or deploy
[Physics-Informed Neural Networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)
(PINNs). It combines an audit of the current library against PINN requirements
with a survey of the relevant literature.

## Verdict

**TinyMind now ships the load-bearing PINN primitive — forward-mode automatic
differentiation of a field with respect to its input coordinates — for every
value type, fixed-point included.** This was originally the one capability gap
(backpropagation computes `∂Loss/∂weights`, the wrong derivative for a PDE
residual). It is closed by `cpp/dual.hpp` and `cpp/dualActivations.hpp`, and
demonstrated end-to-end by `examples/pinn_heat1d/`.

What remains for fully on-device *training* is composing those input
derivatives with derivatives of the residual w.r.t. the **weights**; inference
of a host-trained PINN already works through the existing Q-format pipeline.

> **Status (update):** Sections 1–2 below describe the original gap and its
> resolution. The dual-number machinery they call for is now in the tree and
> tested (`unit_test/dual/`, plus a freestanding `Dual<QValue>` check in
> `unit_test/embedded/`). The "Concrete next steps" section marks what is done
> versus what is left.

## 1. What a PINN requires, and where TinyMind stands

A PINN's loss is the residual of a partial differential equation: a
differential operator applied to the network output `u(x, y, z, t)` with
respect to its **input coordinates**, frequently to second order or higher
(for example `‖∂²u/∂x² − f‖²` for a diffusion equation). Solving a PDE with a
PINN therefore fundamentally requires derivatives of the network output with
respect to its inputs.

| PINN requirement | TinyMind status | Notes |
|---|---|---|
| Differentiate output w.r.t. **inputs** | **Present** (`cpp/dual.hpp`) | Forward-mode `Dual<ValueType>`; `∂u/∂x` in one pass. Was the original gap. |
| 2nd- and higher-order input derivatives | **Present** (nested `Dual`) | `Dual<Dual<…>>` gives `∂²u/∂x²` etc.; verified for polynomials and `tanh`. |
| Smooth activation (C² or better) | **Present** | `tanh`, `sigmoid`, `ELU`, `GELU` with analytic derivatives; `tanh(Dual)`/`sigmoid(Dual)` overloads in `cpp/dualActivations.hpp`. `ReLU` unsuitable (2nd derivative ≡ 0). |
| Small network sizes | **Present** | PINNs are typically small MLPs; TinyMind targets embedded-scale models. |
| Fixed-point / int8 inference | **Present** | Q-format and int8 pipelines run a plain forward pass of `u(x, t)`. |
| Residual-loss training (∂residual/∂weights) | **Not yet** | Composing input-derivative AD with weight gradients; see next steps. |

The takeaway: TinyMind now ships the correct **activation family**, the correct
**size class**, *and* the input-space differentiation operator. The remaining
work is the training loop, not the derivatives.

## 2. The missing piece: forward-mode / Taylor-mode autodiff

The standard and efficient way to obtain a PINN's input derivatives is
**forward-mode automatic differentiation**, implemented with **dual numbers**:

```
a + bε   with   ε² = 0   ⟹   f(a + ε) = f(a) + ε·f′(a)
```

Forward-mode cost scales with the **input dimension**, and PINNs have only
about 1–4 inputs (`x, y, z, t`), so it is the natural choice. It is realized
purely by operator overloading on a dual-number type — a clean fit for a
header-only C++ template library.

- **Higher orders** (for second-order PDEs) come from **Taylor-mode AD**, a
  generalization that carries higher polynomial terms
  (`f(a + ε) = f(a) + ε·f′(a) + ½ε²·f″(a) + …`). Naive nesting of first-order
  AD blows up exponentially in the differentiation order; Taylor-mode avoids
  that.
- **Header-only C++ is proven feasible.**
  [Boost.Math Autodiff](https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/autodiff.html)
  is exactly this — forward-mode, Taylor-series basis, Nth derivative. TinyMind's
  test Makefiles already set `BOOST_HOME`.
- **State-of-the-art PINNs already use it.** Separable PINNs (SPINN) use
  forward-mode AD for the residual; STDE uses Taylor-mode for high-order
  operators.

### Fit to TinyMind's design

A `Dual<ValueType>` type, templated the same way as `QValue`, slots directly
into TinyMind's policy-based design. Because the activation policies already
expose analytic derivatives, propagating a dual number through the network by
the chain rule is straightforward.

**This is now implemented.** `cpp/dual.hpp` provides `Dual<ValueType>` (value +
derivative, sum/product/quotient rules, a `chainRule` hook) built purely from
the value type's arithmetic, so it works identically for `float`/`double` and
`QValue` — and is freestanding-clean (no `<cmath>`/STD/FLOAT required, so
fixed-point input derivatives run on an MCU). `cpp/dualActivations.hpp` adds
`tanh(Dual)` / `sigmoid(Dual)`, with a `DualScalarActivation<V>` policy that
isolates the one type-specific step (LUT for `QValue`, `std::tanh` for
`double`) and recurses through nested duals for higher-order derivatives.

## 3. Precision: the real risk

PINN training is precision- and conditioning-sensitive. Published evidence
shows that even **FP32** — far above int8 or fixed-point — can cause the L-BFGS
optimizer to prematurely satisfy its convergence test and freeze the network in
a spurious "failure phase" where solution error stays large.

Two important boundaries on that evidence (claims that did **not** survive
adversarial verification during this research):

- The claim that FP64 *eliminates* PINN failure modes was **refuted**.
- The claim that precision is the *cause* of failures (rather than optimization
  difficulty) was **refuted**.

So treat "low precision hurts PINN training" as well-supported *directionally*,
but not as "more precision is a guaranteed fix."

**Largest open gap in the literature:** no published benchmark of int8 or
fixed-point PINN accuracy — training *or* inference — was found. The viability
of TinyMind's Q-format pipeline for PINN *inference* is therefore plausible but
**empirically unproven**. This is a genuine research opportunity, not a settled
result.

## 4. Realistic deployment pattern (two stages)

```
Stage 1 — HOST, float:
    PyTorch (or similar) + autograd over inputs  →  PDE residual loss  →  train PINN
    export weights

Stage 2 — MCU, Q-format:
    TinyMind forward pass  u(x, t)  =  inference only
```

- **Works today (verified):** Stage 2 is a plain forward pass, matching
  TinyMind's documented "float training on host, post-training Q-format
  conversion, inference-only on MCU" workflow. `examples/pinn_heat1d/` evaluates
  the same field in float and in Q16.16 fixed point and confirms the inference
  values agree to ~1.7e-3 — no autodiff needed at inference, since deploying a
  trained PINN is just evaluating `u(x, t)`. **This is the primary use case.**
- **Now possible:** the residual *itself* (input derivatives) can be computed
  on-device, in float or fixed-point, via `Dual` — so residual monitoring and
  adaptive collocation sampling no longer require host autograd. Full on-device
  *training* still needs the residual's gradient w.r.t. the weights.

## 5. The KAN angle (promising, under-explored)

TinyMind already ships **B-spline / KAN layers with derivatives**
(`cpp/bspline.hpp`). Physics-informed KANs (PIKANs) are well-suited to small
models. An open question worth probing: **can TinyMind's existing
spline-derivative machinery be repurposed to compute input-coordinate
derivatives for the PDE residual** (rather than weight gradients)? If so, it
could be a shortcut to PINN support via the KAN path. This branch is currently
unverified.

## Concrete next steps for TinyMind

Done:

1. ✅ `Dual<ValueType>` forward-mode type — header-only, value-type-generic,
   freestanding-clean (`cpp/dual.hpp`), validated for `double` and `QValue`.
2. ✅ `tanh(Dual)` / `sigmoid(Dual)` activation overloads (`cpp/dualActivations.hpp`),
   giving `∂u/∂x` directly through a smooth field.
3. ✅ Higher-order derivatives via nested `Dual<Dual<…>>` (`∂²u/∂x²`), verified
   against the analytic `tanh''`.
5. ✅ 1-D exemplar (`examples/pinn_heat1d/`): the heat residual `u_t − ν·u_xx`
   by forward-mode AD — residual ≈ 0 on an exact solution, and autodiff
   derivatives match finite differences on a nonlinear `tanh` field.

Remaining:

4. **Measure the precision unknown:** does fixed-point error in the dual
   coefficients degrade higher-order input derivatives enough to matter? The
   `Dual<QValue>` path is mechanically correct (`unit_test/dual/`); its
   accuracy on a real residual is not yet benchmarked.
6. **On-device training:** compose the input-derivative AD with the residual's
   gradient w.r.t. the weights (reverse-over-forward), then a host-train →
   Q-format-infer comparison for a trained field.
7. **Ergonomic Taylor-mode:** nested `Dual` works but scales awkwardly past 2nd
   order; a dedicated order-N Taylor type would be cleaner for high-order PDEs.

## Caveats and sources

The forward-mode and Taylor-mode AD mathematics is settled textbook material
(high confidence). The PINN speedup figures (SPINN, STDE) are best-case results
for specific PDEs, not universal constants. The precision finding rests on a
single 2025 source whose strongest implications were refuted. **No
quantized-PINN benchmark exists**, so the int8-inference conclusion here is
inferred from general PINN precision-sensitivity plus TinyMind's design, not
measured.

Primary sources:

- Cuomo et al., "Scientific Machine Learning through PINNs," *J. Sci. Comput.*
  2022 — <https://link.springer.com/article/10.1007/s10915-022-01939-z>
- Dashtbayaz et al., IJCAI 2024 — <https://arxiv.org/abs/2405.01680>
- Cho et al., "Separable PINN" (SPINN), NeurIPS 2023 —
  <https://arxiv.org/pdf/2306.15969>
- Shi et al., "Stochastic Taylor Derivative Estimator" (STDE), NeurIPS 2024 —
  <https://arxiv.org/pdf/2412.00088>
- Bettencourt et al., "Taylor-mode AD" (JAX), NeurIPS 2019 workshop —
  <https://openreview.net/pdf?id=SkxEF3FNPH>
- Boost.Math Autodiff —
  <https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/autodiff.html>
- "FP64 is All You Need," NeurIPS 2025 — <https://arxiv.org/abs/2505.10949>
