---
title: int8 Encoder-Decoder (seq2seq)
parent: Examples
nav_order: 62
layout: default
---

# int8 Encoder-Decoder (seq2seq)

The encoder examples produced a memory tensor; this one adds the **decoder** that consumes it, completing an int8 encoder-decoder (seq2seq) transformer. The decoder introduces causal self-attention, cross-attention to the encoder memory, and an autoregressive KV cache — all pure integer.

## How it works

- Pipeline (`Vocab=16, SEnc=6, SDec=6, E=8, F=16`): a source stream runs through one linear-attention **encoder block** (`QLayerNorm1D` → `QAttention1D` → `QAdd`; `QLayerNorm1D` → `QDense`→`qrelu`→`QDense` → `QAdd`) to produce a `[SEnc, E]` memory. A target stream runs through one **decoder block**: `QLayerNorm1D` → `QCausalAttention1D` (causal self) → `QAdd`; `QLayerNorm1D` → `QCrossAttention1D` (to memory) → `QAdd`; `QLayerNorm1D` → `QDense`→`qrelu`→`QDense` → `QAdd`.
- `QCausalAttention1D` (`cpp/qcausalattention1d.hpp`) is causal linear self-attention: position `t` attends only to `s <= t`. With a linear (ReLU-kernel) feature map the causal mask collapses into a fixed `E × E` running KV matrix (`QLinearKVState`, `cpp/qkvcache.hpp`), so autoregressive decode is **O(1) memory per token** — there is no growing cache.
- `QCrossAttention1D` (`cpp/qcrossattention.hpp`) reads two different affine grids — Q from the decoder hidden state, K/V from the encoder memory — so it carries a separate `q_input_zero_point` and `kv_input_zero_point`, with the matching input scales baked into the per-projection requantizers. The encoder K/V collapse into a single `E × E` matrix at prefill; each decoder query is one `Q' · KV`.
- **The headline check:** the int8 causal self-attention is run two ways and asserted byte-identical — one full-sequence `forward()` (training-time teacher forcing) versus `step()` looped token-by-token over the running KV state (inference-time autoregressive decode). The report prints `KV-cache incremental decode == full-sequence: YES (byte-identical)`. That equivalence is what lets a model trained on full sequences deploy as a streaming decoder on an MCU.
- End-to-end error is ~0.7% of output range (pass threshold 40%) for the full encoder + decoder, with no QAT and no cross-layer equalization.

Linear (ReLU-kernel) attention everywhere keeps the whole pipeline freestanding-portable — no softmax/exp LUT. Softmax variants (`QCausalAttentionSoftmax1D`, `QCrossAttentionSoftmax1D`) drop in with an exp LUT from `buildQSoftmaxExpLUT`; both flavors are covered by the quantization unit tests.

## Build and run

```bash
cd examples/seq2seq_int8
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host calibration). Extra target:

- `make golden` — int8 byte stream for the bundled test set to `output/seq2seq_int8.golden`

## Output

![int8 vs float parity for the seq2seq transformer]({{ site.baseurl }}/assets/plots/seq2seq_int8.png)

Left panel overlays the float reference against the int8-dequantized decoder output across all 48 elements — the two tracks are nearly indistinguishable. Right panel shows the per-element absolute error, all under ~0.018, consistent with the ~0.7% of output range for this encoder + decoder int8 pipeline.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/seq2seq_int8)
