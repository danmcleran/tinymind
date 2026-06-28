---
title: int8 seq2seq (softmax decoder)
parent: Examples
nav_order: 64
layout: default
---

# int8 seq2seq (softmax decoder)

The softmax-attention counterpart of [int8 Encoder-Decoder (seq2seq)](seq2seq_int8.html). Same encoder-decoder shape and calibration flow, but the decoder runs standard softmax attention instead of the linear ReLU-kernel — so it exercises `QCausalAttentionSoftmax1D` and `QCrossAttentionSoftmax1D` end-to-end.

## How it works

- Same pipeline as the linear variant (`Vocab=16, SEnc=6, SDec=6, E=8, F=16`): a linear-attention encoder block produces the memory; the decoder block is `QLayerNorm1D` → `QCausalAttentionSoftmax1D` (causal self) → `QAdd`; `QLayerNorm1D` → `QCrossAttentionSoftmax1D` (to memory) → `QAdd`; `QLayerNorm1D` → `QDense`→`qrelu`→`QDense` → `QAdd`.
- **Growing KV cache.** Softmax cannot collapse the attended history into a fixed state the way linear attention does, so `QCausalAttentionSoftmax1D` keeps a `QSoftmaxKVCache` that appends one `(K, V)` row per emitted token. The causal mask is the cache length — the score loop runs only over the rows that exist. Cross-attention prefills the encoder K/V once and scores every decoder query against all of them.
- **Softmax in int8** follows the TFLite convention: per-row max subtract, 256-entry exp LUT (`buildQSoftmaxExpLUT`), 1/256 probability grid. The score requantizer folds the `1/sqrt(d_k)` factor via `qAttentionInvSqrt()`. The exp LUT is the extra footprint over the linear stack.
- **Incremental decode == full-sequence**, byte-for-byte: the int8 causal self-attention is run as one full-sequence `forward()` and as `step()` looped token-by-token over the growing cache, and the two int8 streams are asserted identical.
- End-to-end error is ~0.9% of output range (pass threshold 50%, looser than the linear variant because softmax compounds more int8 stages).

## Build and run

```bash
cd examples/seq2seq_softmax_int8
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host calibration). `make golden` writes the int8 byte stream to `output/seq2seq_softmax_int8.golden`.

## Output

![int8 vs float parity for the softmax-decoder seq2seq]({{ site.baseurl }}/assets/plots/seq2seq_softmax_int8.png)

Left panel overlays the float reference against the int8-dequantized decoder output across all 48 elements — nearly indistinguishable. Right panel shows the per-element absolute error, all under ~0.015, consistent with the ~0.9% of output range for this softmax-decoder int8 pipeline.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/seq2seq_softmax_int8)
