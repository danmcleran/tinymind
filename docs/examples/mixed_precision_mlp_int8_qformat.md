---
title: Mixed-Precision MLP (int8 + Q-format)
parent: Examples
nav_order: 41
layout: default
---

# Mixed-Precision MLP (int8 + Q-format)

A hybrid MLP that runs an int8 affine frontend and classifier around a Q8.8 fixed-point hidden tier, connected by the Phase 17 pure-integer qbridge converters — the entire inference path has no floating-point math at runtime.

## How it works

- Pipeline: `QDense` int8 frontend → `qrelu` → `affineToQValueIntBuffer` bridge → Q8.8 dense matvec (int32 accumulator) → `qValueToAffineIntBuffer` bridge → `QDense` int8 classifier. Both bridges use the same gemmlowp Q0.31 `(multiplier, shift)` primitive that `Requantizer` already relies on, so no `<cmath>`, `<type_traits>`, or `float` is needed at runtime.
- Demonstrates the Phase 17 pure-integer mixed-precision qbridge (`cpp/qbridge.hpp`): the `affineToQValueInt` / `qValueToAffineInt` converters let an int8 affine model and a TinyMind Q-format layer interoperate at the deployable shape `QUANTIZATION=1, FLOAT=0, STD=0`. The conversion triples are built host-side once at calibration time by `buildAffineToQValueIntParams` / `buildQValueToAffineIntParams`.
- The int8 logits land around 0.005 max-abs error vs the float reference on the bundled synthetic dataset, well inside the exemplar's tolerance.

## Build and run

```bash
cd examples/mixed_precision_mlp_int8_qformat
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

The Makefile builds hosted (`FLOAT=1 STD=1 QUANTIZATION=1`) for the parity report, but the inference path itself is freestanding-clean for the deployable `FLOAT=0 STD=0` shape. `make bench` writes a CSV cycle/byte report and `make golden` emits the int8 byte stream wired into the integration suite. `apps/import_pytorch/tinymind_import.py` carries `QFormatDense` / `HybridBoundary` descriptors that emit the same `weights.hpp` layout for importing real PyTorch / TensorFlow models.

## Output

![int8 vs float parity for the hybrid int8 + Q-format MLP]({{ site.baseurl }}/assets/plots/mixed_precision_mlp.png)

The left panel overlays the int8-dequantized output on the float reference; the curves track tightly across all four outputs. The right panel's per-element error stays under ~0.0043, showing the round trip through the Q8.8 hidden tier and the two pure-integer bridges loses only a few thousandths versus all-float math.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/mixed_precision_mlp_int8_qformat)
