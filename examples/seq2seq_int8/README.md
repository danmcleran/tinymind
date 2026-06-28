# seq2seq_int8

A full int8 **encoder–decoder (seq2seq) transformer**. This finishes the
transformer arc in TinyMind: the encoder family
([`../transformer_encoder_int8`](../transformer_encoder_int8),
[`../transformer_encoder_stack_int8`](../transformer_encoder_stack_int8))
produced a memory tensor; here a **decoder** consumes it.

```
source ids (SEnc)                      target ids (SDec)
     |                                      |
[QEmbedding]+[QPositional]            [QEmbedding]+[QPositional]
     |                                      |
ENCODER block:                        DECODER block:
  LN -> LinAttn -> Add                  LN -> CausalLinAttn -> Add     (self)
  LN -> FFN    -> Add                   LN -> CrossLinAttn  -> Add     (to memory)
     |                                    LN -> FFN -> Add
memory (SEnc, E) -------------------->    |
                                        output (SDec, E)
```

Every stage runs in pure int8 affine arithmetic. Linear (ReLU-kernel)
attention keeps the whole pipeline freestanding-portable — no softmax/exp LUT.

## What the decoder adds

| Header | Layer | Role |
|--------|-------|------|
| `cpp/qkvcache.hpp` | `QLinearKVState`, `QSoftmaxKVCache` | autoregressive KV state (caller-owned, freestanding POD) |
| `cpp/qcausalattention1d.hpp` | `QCausalAttention1D` | **causal** linear self-attention: `forward()` (teacher forcing) + `step()` (decode) |
| `cpp/qcausalattention_softmax.hpp` | `QCausalAttentionSoftmax1D` | causal softmax self-attention with a growing KV cache |
| `cpp/qcrossattention.hpp` | `QCrossAttention1D`, `QCrossAttentionSoftmax1D` | encoder–decoder cross-attention (linear + softmax) |

The encoder block and the surrounding `QLayerNorm1D` / `QDense` / `QAdd` /
`QEmbedding` / `QPositionalEncoding1D` layers are the same ones the encoder
examples use.

## The headline: O(1) KV-cache incremental decode

The decoder self-attention is **causal** — position `t` attends only to
`s <= t`. For linear attention that causal mask collapses into a fixed
`E × E` running KV matrix (`QLinearKVState`), so autoregressive decoding is
**O(1) memory per token** — there is no growing cache.

The driver runs the int8 causal self-attention two ways and asserts they are
**byte-identical**:

* `forward()` — one full-sequence pass (training-time teacher forcing), and
* `step()` looped token-by-token over the running KV state (inference-time
  autoregressive decode).

That equivalence — printed as `KV-cache incremental decode == full-sequence:
YES (byte-identical)` — is exactly what lets a model trained on full
sequences deploy as a streaming decoder on an MCU.

## Build & run

```bash
make            # debug build
make run        # int8 vs float parity + cache-equivalence report
make golden     # dump int8 codes to *.golden for regression
make plot       # int8/float overlay + per-element error PNG
```

The driver builds the identical encoder+decoder in float, calibrates every
activation tensor on a small synthetic source/target token dataset
(post-training, per-tensor), builds the int8 layers, and prints the
worst-case max-abs error after dequantization (~0.7% of the output range on
the bundled data).

Cross-attention reads two different affine grids — Q from the decoder hidden
state, K/V from the encoder memory — so `QCrossAttention1D` carries a separate
`q_input_zero_point` and `kv_input_zero_point`, with the matching input scales
baked into the per-projection requantizers.

## Knobs

`VOCAB`, `SENC` / `SDEC` (source/target lengths), `E` (model dim), `F` (FFN
inner dim) are `constexpr` at the top of the source. To switch the decoder to
standard softmax attention, swap `QCausalAttention1D` for
`QCausalAttentionSoftmax1D` (self) and `QCrossAttention1D` for
`QCrossAttentionSoftmax1D` (cross) and supply an exp LUT from
`buildQSoftmaxExpLUT` — both flavors are covered by the quantization unit
tests.
