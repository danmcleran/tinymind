# seq2seq_softmax_int8

The **softmax-attention** counterpart of [`../seq2seq_int8`](../seq2seq_int8).
Same int8 encoder-decoder (seq2seq) shape, but the decoder uses standard
softmax attention instead of the linear ReLU-kernel:

```
ENCODER block (linear attn):      DECODER block (softmax attn):
  LN -> LinAttn -> Add              LN -> CausalSoftmaxAttn -> Add  (self)
  LN -> FFN    -> Add               LN -> CrossSoftmaxAttn  -> Add  (to memory)
     |                                LN -> FFN -> Add
memory (SEnc, E) ----------------->   |
                                    output (SDec, E)
```

## Linear vs softmax: what changes

The linear variant collapses the decoder's attended history into a fixed
`E × E` KV state (O(1) per token). Softmax **cannot** — it needs every past
key and value — so `QCausalAttentionSoftmax1D` keeps a **growing**
`QSoftmaxKVCache`: each emitted token appends one `(K, V)` row, and the causal
mask is simply the cache length (the score loop only runs over rows that
exist). Cross-attention (`QCrossAttentionSoftmax1D`) prefills the encoder K/V
once and scores every decoder query against all of them.

Both softmax layers follow the TFLite int8 convention: per-row max subtract,
256-entry exp LUT (`buildQSoftmaxExpLUT`), 1/256 probability grid. The score
requantizer folds the `1/sqrt(d_k)` factor via `qAttentionInvSqrt()`. The exp
LUT is the extra footprint over the linear variant.

## The incremental-decode check still holds

As in the linear example, the int8 causal self-attention is run two ways and
asserted **byte-identical**:

* `forward()` — one full-sequence pass (teacher forcing), and
* `step()` looped token-by-token, appending to the growing cache.

Printed as `growing-cache incremental decode == full-sequence: YES
(byte-identical)`.

## Build & run

```bash
make            # debug build
make run        # int8 vs float parity + cache-equivalence report
make golden     # dump int8 codes for regression
make plot       # int8/float overlay + per-element error PNG
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host
calibration). Tolerance is 50% of output range (looser than the linear
variant's 40% — softmax compounds more int8 stages); the bundled run lands at
~0.9% max-abs error vs float.
