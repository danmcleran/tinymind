# import_moe_demo — Mixture-of-Experts importer round-trip

End-to-end proof that the `apps/import_pytorch` importer emits a Mixture-of-Experts
layer the C++ `cpp/qmoe.hpp::QMixtureOfExperts` runtime reproduces faithfully.

## What it does

1. `demo.py` builds a deterministic top-1 MoE in float (4 inputs → 3 experts →
   3 outputs) with numpy — no torch. The router emphasizes a different input
   dimension per expert so routing is input-dependent and all experts fire.
2. It calibrates the model through `import_pytorch_model` (MinMax input
   observer, percentile output observer) and emits `weights.hpp`:
   - router weights (symmetric int8) + int32 router bias — **no requantizer**,
     the router argmaxes over raw int32 logits;
   - per-expert int8 weights, each with its **own** weight scale;
   - one **shared** MoE output scale / zero point.
   It also emits `moe_reference.hpp`: held-out test inputs, the expected
   selected expert, and the float top-1 output.
3. `import_moe_demo.cpp` consumes those headers, rebuilds `QMixtureOfExperts`
   (one `buildRequantizer` per expert into the shared output scale), runs the
   pure-integer forward, and checks:
   - the selected expert matches the float-model argmax (exact), and
   - the dequantized int8 output matches the float reference within tolerance.

## Build & run

```bash
make            # build (debug)
make run        # run the parity check
```

Expected:

```
import_moe_demo parity test (12 samples)
  selected-expert mismatches = 0 / 12
  max |y_int8 - y_float|     = 0.008972
  tolerance 0.050000 : PASS
```

The checked-in `weights.hpp` / `moe_reference.hpp` let the binary build without
numpy. To regenerate them from the float model:

```bash
make regenerate-pytorch   # runs demo.py (needs numpy)
```

## Why per-expert weight scale but shared output scale

Each expert is calibrated to use the full int8 range for its own weights
(per-expert weight scale), which captures the experts' differing dynamic
ranges. The MoE output, however, feeds a single downstream consumer, so all
experts requantize into one shared output scale/zero point — otherwise the
int8 output bytes would mean different real values depending on which expert
fired. Routing is scale-invariant (argmax over raw int32 logits), so the router
needs no requantizer and tolerates a low-precision router.
