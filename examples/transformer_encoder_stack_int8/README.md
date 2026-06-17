# transformer_encoder_stack_int8

A full int8 transformer **encoder stack** — the single-block demo in
[`../transformer_encoder_int8`](../transformer_encoder_int8) grown into an
end-to-end model that starts from token ids:

```
token ids (S)
    |
[QEmbedding]            gather int8 rows from a [Vocab, E] table   (qembedding.hpp)
    |
[QPositionalEncoding]   add fixed sinusoidal table via QAdd         (qpositional.hpp)
    |
[EncoderBlock 0]        LN -> LinAttention -> Add ; LN -> Dense->ReLU->Dense -> Add
    |
[EncoderBlock 1]        (NUM_BLOCKS stacked, each grid-chained to the last)
    |
output (S, E)
```

Every stage runs in pure int8 affine arithmetic. Linear (ReLU-kernel)
attention keeps the whole stack freestanding-portable — no softmax/exp LUT.
Swap `QAttention1D` for `QAttentionSoftmax1D` for standard attention.

## New primitives exercised

| Header | Layer | Role |
|--------|-------|------|
| `cpp/qembedding.hpp`   | `QEmbedding`              | token id → int8 embedding gather |
| `cpp/qpositional.hpp`  | `QPositionalEncoding1D`   | fused affine add of a positional table (wraps `QAdd`); ships `sinusoidalPositionalTable()` host generator |

The remaining layers (`QLayerNorm1D`, `QAttention1D`, `QDense`, `QAdd`) are
the same ones the single-block demo uses.

## Build & run

```bash
make            # debug build
make run        # int8 vs float parity report
make golden     # dump int8 codes to *.golden for regression
make plot       # int8/float overlay + per-element error PNG
```

The driver builds the identical stack in float, calibrates every activation
tensor on a small synthetic token dataset (post-training, per-tensor), builds
the int8 layers, and prints the worst-case max-abs error after dequantization.
Block N's output grid feeds block N+1's input grid directly — the same
grid-chaining a real deployment uses.

Tolerance is a loose 40% of the output range: this is embedding + positional
add + `NUM_BLOCKS` of stacked attention/FFN with **no** QAT and **no**
cross-layer equalization. Tighten it through the Phase-15 importer tooling
(`apps/import_pytorch`, `apps/import_onnx`).

## Knobs

`VOCAB`, `S` (sequence length), `E` (embedding dim), `F` (FFN inner dim) and
`NUM_BLOCKS` are `constexpr` at the top of the source. The stack depth is the
interesting one: raise `NUM_BLOCKS` and watch quantization error accumulate
across grid-chained blocks.
