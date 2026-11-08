---
title: MoE Importer Round-Trip
parent: Examples
nav_order: 70
layout: default
---

# MoE Importer Round-Trip

An end-to-end check that the [`apps/import_pytorch`](https://github.com/danmcleran/tinymind/tree/master/apps/import_pytorch) importer emits a Mixture-of-Experts layer that the C++ `cpp/qmoe.hpp::QMixtureOfExperts` runtime reproduces.

## How it works

1. `demo.py` builds a deterministic top-1 MoE in float (4 inputs, 3 experts, 3 outputs) with numpy, no torch. The router emphasizes a different input dimension per expert, so routing depends on the input and every expert fires.
2. It calibrates the model through `import_pytorch_model` (MinMax input observer, percentile output observer) and emits `weights.hpp`:
   - router weights (symmetric int8) and an int32 router bias, with **no requantizer**, because the router argmaxes over raw int32 logits;
   - int8 weights for each expert, each with its **own** weight scale;
   - one **shared** output scale and zero point.

   It also emits `moe_reference.hpp`: held-out test inputs, the expected selected expert, and the float top-1 output.
3. `import_moe_demo.cpp` consumes those headers, rebuilds `QMixtureOfExperts` (one `buildRequantizer` per expert into the shared output scale), runs the pure-integer forward pass, and checks that the selected expert matches the float model's argmax exactly and that the dequantized output matches the float reference within tolerance.

Per-expert weight scales capture the experts' differing dynamic ranges. The output scale is shared because the MoE output feeds one downstream consumer: with a scale per expert, an output byte would mean a different real value depending on which expert fired.

## Build and run

```bash
cd examples/import_moe_demo
make
make run                  # runs the parity check
make regenerate-pytorch   # re-runs demo.py to regenerate the headers (needs numpy)
```

The checked-in `weights.hpp` and `moe_reference.hpp` let the binary build without numpy.

## Output

```
import_moe_demo parity test (12 samples)
  selected-expert mismatches = 0 / 12
  max |y_int8 - y_float|     = 0.008972
  tolerance 0.050000 : PASS
```

See [Mixture of Experts]({{ site.baseurl }}/architectures/mixture-of-experts) for the layer family and [PyTorch → tinymind int8 (importer)]({{ site.baseurl }}/getting-started/pytorch-importer) for the importer flow.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/import_moe_demo)
