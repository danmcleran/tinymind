---
title: int8 Mixture-of-Experts Routing
parent: Examples
nav_order: 69
layout: default
---

# int8 Mixture-of-Experts Routing

A heterogeneous 1D regression that shows what a top-1 Mixture-of-Experts buys on an embedded target. The nonlinear target `f(x) = 0.6·sin(1.5x) + 0.25x` is approximated by **three linear experts**; a tiny linear router splits the input domain into three regimes and picks one expert per input. Deployed in int8 through `cpp/qmoe.hpp::QMixtureOfExperts`.

## How it works

- **Routing.** The router argmaxes over **raw int32 logits**. It needs no requantizer, because argmax does not change under a shared positive scale. The argmax over three lines is their upper envelope, which falls out as three contiguous intervals (boundaries at x = ±1).
- **Experts.** Each expert is the least-squares line fit to the curve inside its regime, with its **own** per-expert weight scale. All experts requantize into one **shared** output scale, so an output byte means the same real value whichever expert fired.
- **Compute is not memory.** All three experts are resident; exactly one runs per inference. Active compute scales with the number of experts run, and resident memory scales with the number of experts held.

`make bench` prints the accounting:

```
MoE compute/memory accounting (top-1 routing)
  experts resident         : 3
  experts run per inference : 1
  resident params (bytes)   : 12  (router + all experts, all must be in flash)
  active params per call    : 8  (router + selected expert)
  MACs per inference        : 4  (router 3 + 1 expert 1)
  MACs if all experts ran   : 6  (dense-MoE upper bound)
  routing over domain       : e0=80 e1=81 e2=80
```

The gap between prediction and target in the plot is the three-line approximation, not int8 loss: one output LSB is about 0.008, and the kinks near the regime boundaries are where a piecewise-linear fit of a curve falls short.

## Build and run

```bash
cd examples/moe_regimes_int8
make release
make run        # writes output/moe_regimes_int8.csv (x, target, prediction, expert)
make bench      # prints the compute/memory accounting
make golden     # prints golden int8 bytes for a fixed probe set
make plot       # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

The float router and expert lines are fit inside the binary, so the example needs no numpy. In a real workflow the router and experts are trained offline and loaded from an importer-emitted `weights.hpp`; see [MoE importer round-trip]({{ site.baseurl }}/examples/import_moe_demo).

## Output

![int8 MoE regime routing]({{ site.baseurl }}/assets/plots/moe_regimes_int8.png)

Each color block is the output of a single expert, so the colors *are* the routing map.

See [Mixture of Experts]({{ site.baseurl }}/architectures/mixture-of-experts) for the layer family.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/moe_regimes_int8)
