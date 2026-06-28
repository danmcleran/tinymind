---
title: Transformer Decoder (seq2seq)
layout: default
parent: Architectures
nav_order: 10
---

# Transformer Decoder (seq2seq)

> **Real-world use:** an offline voice-command device turns a recognized phrase into a structured control sequence ("set thermostat to 21 and arm the alarm"). The decoder generates the output tokens autoregressively against a fixed-size KV state, so memory stays flat no matter how long the command — the whole sequence-to-sequence step runs on-device with nothing sent off-chip.

TinyMind's encoder layers (linear / softmax self-attention, embedding,
positional encoding) produce a memory tensor. The **decoder** family consumes
it, completing an int8 encoder-decoder (seq2seq) transformer. It adds three
things the encoder does not need: a **causal mask**, an **autoregressive KV
cache**, and **cross-attention** to the encoder memory.

All of it is pure integer at inference time. The linear (ReLU-kernel) flavors
are freestanding-safe (no LUT); the softmax flavors reuse the same host-built
exp LUT as the rest of the int8 family.

| Header | Layer | Role |
|--------|-------|------|
| `cpp/qkvcache.hpp` | `QLinearKVState`, `QSoftmaxKVCache` | caller-owned autoregressive KV state (freestanding POD) |
| `cpp/qcausalattention1d.hpp` | `QCausalAttention1D` | causal linear self-attention |
| `cpp/qcausalattention_softmax.hpp` | `QCausalAttentionSoftmax1D` | causal softmax self-attention |
| `cpp/qcrossattention.hpp` | `QCrossAttention1D`, `QCrossAttentionSoftmax1D` | cross-attention (linear + softmax) |

## Causal linear attention collapses the KV cache to O(1)

A decoder is autoregressive: position `t` may attend only to positions
`s <= t` (the causal mask). For **standard softmax** attention that means
keeping every past key and value around — the familiar KV cache that grows one
row per emitted token (`QSoftmaxKVCache`, memory `O(MaxSeq · P)`).

For **linear** attention the causal mask is something much cheaper. Linear
attention reorders the product so the keys and values meet first:

```
y[t] = Q'[t] · ( sum_{s <= t} K'[s]^T V[s] )
```

The bracketed term is a running prefix sum — a fixed `P × P` matrix that simply
accumulates one outer product per step. So the entire attended history
collapses into a constant-size `E × E` int32 state (`QLinearKVState`), and
autoregressive decode is **O(1) memory per token**: there is no growing cache.
On a target with kilobytes of SRAM that is the difference between a decoder
that fits and one that does not.

## One kernel, two entry points

Both causal layers expose the same pair of methods over one shared token
kernel:

```cpp
// Inference: fold one new token into the cache, emit its output row.
//   O(P^2) work, O(1) extra memory per token.
void step(const InputType* x_row, KVState& state, /* scratch */, OutputType* o_row) const;

// Training-time teacher forcing: reset the state and run step() across a block.
//   Byte-identical to S successive step() calls.
void forward(const InputType* input, KVState& state, /* scratch */, OutputType* output) const;
```

Because `forward()` is literally `state.reset()` followed by a loop of
`step()`, a model run as one full-sequence pass during calibration produces the
**exact same int8 stream** as the token-by-token decode at deployment. The
[`seq2seq_int8`]({{ site.baseurl }}/examples/seq2seq_int8) example asserts this
byte-for-byte, which is the property that makes "train on full sequences,
deploy as a streaming decoder" valid rather than approximate.

## Cross-attention reads two grids

Cross-attention is where the decoder looks at the encoder. Unlike
self-attention, the queries and the keys/values come from **different
tensors** with different affine grids — Q from the decoder hidden state, K/V
from the (constant) encoder memory — and there is no causal mask: every
decoder position sees the whole encoder output.

`QCrossAttention1D` / `QCrossAttentionSoftmax1D` therefore carry two input zero
points, `q_input_zero_point` and `kv_input_zero_point`, with the matching input
scales folded into the per-projection requantizers. The encoder K/V depend only
on the constant memory, so they are **prefilled once** (the linear flavor
collapses them into a single `E × E` matrix; the softmax flavor fills a
`QSoftmaxKVCache`) and then read on every decoder query — the per-token decode
cost is just the Q projection and one attention read.

## A decoder block

The blocks are caller-composed from the layer family, matching how the encoder
examples build an encoder block:

```
LN -> QCausalAttention1D  -> QAdd (skip)     # masked self-attention
LN -> QCrossAttention1D   -> QAdd (skip)     # attend to encoder memory
LN -> QDense -> qrelu -> QDense -> QAdd (skip) # position-wise FFN
```

Layer norm, the residual adds, and the FFN are position-independent, so the
only sequence-coupled component is the causal self-attention — which is exactly
the part the KV state makes incremental. See the
[`seq2seq_int8`]({{ site.baseurl }}/examples/seq2seq_int8) example for the full
encoder + decoder pipeline, calibration, and the incremental-decode equivalence
check.
