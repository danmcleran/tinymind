# import_demo (Phase 15 importer end-to-end)

End-to-end demonstration of the Phase 15 PyTorch -> TinyMind int8 importer
path. Two complementary entry points share this directory:

  * **`import_demo.cpp`** -- the host binary built by `make`. It carries a
    deterministic 3-8-4-2 MLP with hardcoded float weights, drives a
    64-sample synthetic calibration set through the Phase 15 observers
    (`RangeObserver`, `PercentileObserver`, `KLDivergenceObserver`) plus
    the Cross-Layer Equalization pass, then runs both the float reference
    and the pure-integer int8 forward and reports max-abs error. Stands
    alone; no Python / PyTorch needed.
  * **`demo.py`** -- production importer flow. Trains the same shape MLP
    in PyTorch, pulls numpy weights from `torch.state_dict`, then drives
    `apps/import_pytorch/tinymind_import.import_pytorch_model` to emit a
    real `weights.hpp`. Run via `make regenerate-pytorch` (requires
    torch + numpy via pyenv).

## Build & run

```bash
make clean && make && make run
```

Expected output:

```
CLE float drift (ReLU model): 0.0...e-05
ranges:
  input  scale=...  zp=...
  h1     scale=...  zp=...
  h2     scale=...  zp=...  (percentile [...])
  logit  scale=...  zp=...  (kl threshold=...)

import_demo parity test (16 samples)
  max |y_int8 - y_float| = 0.0...
  tolerance 0.08 : PASS
```

Exit status 0 on PASS, 1 on FAIL.

## What this exercises

| Phase 15 deliverable                | Where used                                  |
|-------------------------------------|---------------------------------------------|
| `PercentileObserver`                | hidden-2 activation calibration             |
| `KLDivergenceObserver`              | logit calibration (heavy-tail)              |
| `crossLayerEqualizeDense`           | fc1 / fc2 channel rebalancing pre-quant     |
| `apps/import_pytorch/tinymind_import` | `demo.py` (production flow)               |
| `foldBatchNorm` (Phase 11)          | unused in this MLP (no BN layers)           |

For a Conv+BN example see `examples/resnet_block_int8/` which already
exercises `foldBatchNorm`.

## Notes

The C++ binary engineers a 100:1 per-channel weight imbalance between fc1
and fc2 to force CLE to do real work; the printed `CLE float drift` line
confirms that the equalization is bit-equivalent to the original ReLU
model (positive-homogeneous scaling commutes with ReLU). The Phase 15
`crossLayerEqualizeDense` test case in `unit_test/quantization/` locks
this invariant.
