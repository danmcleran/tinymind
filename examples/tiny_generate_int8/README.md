# tiny_generate_int8

A tiny **autoregressive text generation** demo in pure int8 — a decoder-only
(GPT-style) nano language model that generates one token at a time. This is
the decoder family ([`qcausalattention1d.hpp`](../../cpp/qcausalattention1d.hpp),
[`qkvcache.hpp`](../../cpp/qkvcache.hpp)) in its real loop, the natural sequel
to [`../seq2seq_int8`](../seq2seq_int8).

```
token[t] --[QEmbedding]+[pos]--> x
   x --LN--> [QCausalAttention1D.step] --+Add(skip)-->
      --LN--> [QDense->qrelu->QDense] --+Add(skip)--> h
   h --[QDense head]--> logits (Vocab) --argmax--> token[t+1]
                                                       |
   <---------------------- feed back -------------------'
```

## O(1) attention memory — the point

The decode loop calls `QCausalAttention1D::step()` — **never** `forward()`.
The entire attended history lives in a fixed `E × E` `QLinearKVState`, so the
model generates an arbitrarily long sequence in constant attention memory: the
state is the same size after token 100 as after token 1. There is no growing
KV cache. That is exactly what makes autoregressive decoding fit on an MCU.

## int8 reproduces float

Greedy (argmax) decoding is deterministic and `argmax` is invariant under a
positive affine map, so an int8 logit grid reproduces the float reference's
arg-max whenever the top-1 margin clears the quantization step. The driver runs
the identical model in float and asserts the int8 greedy decode picks the
**same tokens** — 48/48 on the bundled prompts.

## The "+1 counter" wiring

With random untrained weights a tiny LM collapses to a constant token, which
is a dull demo. So the nano-LM is hand-wired as a deterministic **+1 counter**:
orthogonal one-hot token embeddings carried by the residual path and a
shift-by-one readout head, so greedy decode emits `token+1 (mod VOCAB)` each
step. The output is an interpretable sawtooth — each prompt continues the
counter from its last seed token — while the attention / FFN / KV machinery
still runs underneath (it perturbs a dominant residual). The point of the
example is the *decode mechanism*, not a trained model.

## Build & run

```bash
make            # debug build
make run        # generate from the seed prompts, float vs int8 token match
make golden     # dump generated int8 token streams for regression
make plot       # generated-sequence overlay PNG (float line + int8 markers)
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host
calibration).

## Knobs

`VOCAB` (= `E`, kept equal for orthogonal embeddings), `F` (FFN inner dim),
`PROMPT` (seed length), `GEN` (tokens to generate) are `constexpr` at the top
of the source. Raise `GEN` arbitrarily — the KV state size does not change,
which is the whole demonstration. To drive a *trained* model instead of the
counter wiring, import int8 weights via `apps/import_pytorch` /
`apps/import_onnx` and drop them into the same layer instances.
