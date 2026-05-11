# transformer_encoder_int8

Phase 13 demonstration: int8 transformer encoder block.

## What it shows

A single post-LayerNorm-style encoder block stitched together from
Phase 11 normalization, Phase 10 composition, and Phase 13 attention
primitives:

```
input  (S=4, E=8)
   │
   ├──────────────► QLayerNorm1D ──► QAttention1D (linear, ReLU-kernel)
   │                                          │
   └──────────────────► QAdd ◄────────────────┘
                          │
                          ├────────────────► QLayerNorm1D
                          │                       │
                          │                  QDense (E -> F=16) ── qrelu
                          │                       │
                          │                  QDense (F -> E)
                          │                       │
                          └─────► QAdd ◄──────────┘
                                    │
                                output (S, E)
```

`QAttention1D` is the Phase 13 linear-attention primitive (ReLU kernel
feature map instead of softmax). Drop in `QAttentionSoftmax1D` for the
standard variant — calibration adds a softmax exp LUT and a score
requantizer that folds the 1 / sqrt(d_k) factor. `QMultiHeadLinearAttention1D`
wraps `NumHeads` heads if you need multi-head attention.

## How it runs

The driver hand-builds deterministic weights for two LayerNorms, three
attention projections (W_q, W_k, W_v), and two FFN dense layers, runs the
block in float over an 8-sample synthetic dataset to collect every
intermediate tensor's range, calibrates per-tensor activation params
(`computeAffineParamsAsymmetric`) and per-tensor symmetric weight scales,
quantizes weights and biases, builds Requantizers, then runs the block
end-to-end on int8. Output is dequantized and compared against the float
reference.

ReLU on the Q'/K' projections inside `QAttention1D` and on the FFN's
post-Dense1 activations is folded into the upstream `Requantizer` by
raising `qmin` to the destination tensor's `zero_point` — the same
fused-clamp pattern used elsewhere in the Q* family.

## Numbers

After `make run`, with the bundled synthetic dataset:

- block end-to-end max-abs error: ~2 % of output dynamic range (8 samples)

Pass threshold is 40 % of the dynamic range — generous because this is a
six-stage int8 chain (LN, Attn, Add, LN, FFN, Add) with no QAT and no
cross-layer equalization. Phase 15 (importer + calibration upgrades)
tightens this.

## Files

- `transformer_encoder_int8.cpp` — driver. Self-contained; no external
  weight file. Uses `QLayerNorm1D`, `QAttention1D`, `QAdd`, `QDense`,
  `qrelu`, plus calibration helpers from `cpp/include/qcalibration.hpp`.
- `Makefile` — builds with `TINYMIND_ENABLE_FLOAT=1 STD=1 QUANTIZATION=1`.

## Build and run

```sh
make clean && make && make run
```

## Swapping in softmax attention or QFFT

- **Softmax attention.** Replace `QAttention1D` with `QAttentionSoftmax1D`,
  add a 256-entry `int32_t exp_lut[]` built by `buildQSoftmaxExpLUT`, and
  fold the `1 / sqrt(P)` factor into the score requantizer via
  `qAttentionInvSqrt(P)`. Score / attention scratch buffers grow to
  `S * S`.

- **Audio front-end.** `QFFT1D<N>` accepts a caller-owned Q1.15 twiddle
  table (`buildQFFTTwiddles`) and an int16 workspace. Pipe its
  magnitude-squared output through a per-bin `Requantizer` to land on the
  int8 grid the encoder expects.
