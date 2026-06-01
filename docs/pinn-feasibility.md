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

**Today, TinyMind is a viable _inference_ target for a trained PINN, but it
cannot _train_ one.** A single capability gap is responsible: TinyMind has no
automatic differentiation of a network's output with respect to its **input
coordinates**. Everything else a PINN needs — smooth activations, small model
sizes, fixed-point inference — TinyMind already provides.

## 1. What a PINN requires, and where TinyMind stands

A PINN's loss is the residual of a partial differential equation: a
differential operator applied to the network output `u(x, y, z, t)` with
respect to its **input coordinates**, frequently to second order or higher
(for example `‖∂²u/∂x² − f‖²` for a diffusion equation). Solving a PDE with a
PINN therefore fundamentally requires derivatives of the network output with
respect to its inputs.

| PINN requirement | TinyMind status | Notes |
|---|---|---|
| Differentiate output w.r.t. **inputs** | **Missing** | Backpropagation computes `∂Loss/∂weights` only — the wrong derivative for a PDE residual. |
| 2nd- and higher-order input derivatives | **Missing** | Required for diffusion, Poisson, wave equations, etc. |
| Smooth activation (C² or better) | **Present** | `tanh`, `sigmoid`, `ELU`, `GELU`, all with analytic derivatives. `tanh` is the de-facto PINN default; `ReLU` is unsuitable because its second derivative is identically zero. |
| Small network sizes | **Present** | PINNs are typically small MLPs; TinyMind targets embedded-scale models. |
| Fixed-point / int8 inference | **Present** | Q-format and int8 pipelines run a plain forward pass of `u(x, t)`. |

The takeaway: TinyMind already ships the correct **activation family** and the
correct **size class**. The only missing piece is one specific operator —
differentiation in input space.

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
the chain rule is straightforward. This is the natural extension path to PINN
support.

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

- **Works today:** Stage 2 is a plain forward pass, matching TinyMind's
  documented "float training on host, post-training Q-format conversion,
  inference-only on MCU" workflow.
- **Breaks:** anything that needs the residual on-device — on-device
  fine-tuning, residual monitoring, adaptive collocation sampling — because
  those require the input-derivative AD TinyMind lacks.

## 5. The KAN angle (promising, under-explored)

TinyMind already ships **B-spline / KAN layers with derivatives**
(`cpp/bspline.hpp`). Physics-informed KANs (PIKANs) are well-suited to small
models. An open question worth probing: **can TinyMind's existing
spline-derivative machinery be repurposed to compute input-coordinate
derivatives for the PDE residual** (rather than weight gradients)? If so, it
could be a shortcut to PINN support via the KAN path. This branch is currently
unverified.

## Concrete next steps for TinyMind

1. Add a `Dual<ValueType>` forward-mode type — header-only, templated on the
   value type. Validate against `float`/`double` first.
2. Wire the activation policies' analytic derivatives into dual propagation to
   obtain `∂u/∂x` directly.
3. Add Taylor-mode (order-N dual) for second-order PDEs.
4. **Measure the unknown:** does fixed-point error in the dual/Taylor
   coefficients degrade higher-order input derivatives enough to break
   residual-based on-device fine-tuning?
5. Ship a 1-D exemplar (e.g. the heat equation `u_t = ν·u_xx`, or Burgers'):
   train on host → run Q-format inference on device → report residual error vs.
   the float reference.

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
