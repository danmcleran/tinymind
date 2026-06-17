# transformer_encoder_stack_softmax_int8

The [`../transformer_encoder_stack_int8`](../transformer_encoder_stack_int8)
pipeline with **standard (softmax) self-attention** in place of linear
(ReLU-kernel) attention:

```
token ids (S)
    |
[QEmbedding]            int8 gather from [Vocab, E] table   (qembedding.hpp)
    |
[QPositionalEncoding]   add fixed sinusoidal table          (qpositional.hpp)
    |
[EncoderBlock 0..N-1]   LN -> SoftmaxAttention -> Add ; LN -> Dense->ReLU->Dense -> Add
    |
output (S, E)
```

## Softmax attention in int8

`QAttentionSoftmax1D` follows the TFLite int8 convention:

1. Project Q, K, V (no ReLU).
2. Score = (Q · Kᵀ) / √P, requantized onto an int8 **score grid**. The 1/√P
   factor is folded into `score_requantizer` at calibration time.
3. Per row: subtract the row max, look exp up in a **256-entry int32 LUT**,
   normalize to the 1/256 probability grid at zero_point −128.
4. Output = probabilities · V, requantized onto the block's attention grid.

The exp LUT (`buildQSoftmaxExpLUT`) is the extra footprint this variant pays
over the linear-attention stack — 1 KiB of int32 per distinct score grid, in
flash on a freestanding target. That is the trade: true softmax weighting for
an LUT and a few more requantize stages.

## Build & run

```bash
make            # debug build
make run        # int8 vs float parity report
make golden     # dump int8 codes for regression
make plot       # int8/float overlay + per-element error PNG
```

Tolerance is 50% of the output range (looser than the linear stack's 40%):
softmax compounds more int8 stages — score grid, LUT, normalize — so the
post-training noise floor is higher. Phase-15 importer tooling tightens it.

## Knobs

`VOCAB`, `S`, `E` (= projection dim P), `F`, `NUM_BLOCKS` are `constexpr` at
the top of the source. Swap `QAttentionSoftmax1D` back to `QAttention1D` (and
drop the LUT / score grid) to recover the linear-attention sibling.
