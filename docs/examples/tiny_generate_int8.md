---
title: int8 Autoregressive Generation
parent: Examples
nav_order: 63
layout: default
---

# int8 Autoregressive Generation

A decoder-only (GPT-style) **nano language model** that generates text one token at a time, entirely in int8. This is the decoder family in its real loop: `QCausalAttention1D::step()` over a fixed KV state, an output head, and greedy (argmax) decoding that feeds each predicted token back as the next input.

## How it works

- Per step: `QEmbedding` gathers the current token, a `QAdd` injects the positional row, then `QLayerNorm1D` → `QCausalAttention1D::step()` → `QAdd` skip → `QLayerNorm1D` → `QDense`→`qrelu`→`QDense` → `QAdd` skip → `QDense` head → `argmax` over the int8 logits picks the next token.
- **O(1) attention memory.** The decode loop never calls `forward()` — only `step()`. The entire attended history lives in a fixed `E × E` `QLinearKVState` (`cpp/qkvcache.hpp`), so the model generates an arbitrarily long sequence with the same state size after token 100 as after token 1. There is no growing KV cache — exactly what makes autoregressive decoding fit on an MCU.
- **int8 reproduces float.** `argmax` is invariant under a positive affine map, so an int8 logit grid reproduces the float reference's arg-max whenever the top-1 margin clears the quantization step. The driver runs the identical model in float and asserts the int8 greedy decode picks the same tokens — 48/48 on the bundled run.
- The nano-LM is hand-wired as a deterministic **"+1 counter"**: orthogonal one-hot token embeddings carried by the residual path and a shift-by-one readout head, so greedy decode emits `token+1 (mod VOCAB)` each step. That makes the output an interpretable sawtooth rather than a collapsed constant, while the attention/FFN/KV machinery still runs underneath.

## Build and run

```bash
cd examples/tiny_generate_int8
make release
make run
make plot      # needs matplotlib; a venv/pyenv works if it is not already in your Python
```

Built with `-DTINYMIND_ENABLE_QUANTIZATION=1` (plus `FLOAT=1 STD=1` for host calibration). Extra target:

- `make golden` — generated int8 token streams for the bundled prompts to `output/tiny_generate_int8.golden`

## Output

![int8 greedy decode vs float for the nano-LM]({{ site.baseurl }}/assets/plots/tiny_generate_int8.png)

Each panel is one seed prompt's generated sequence: the blue step line is the float reference, the orange ×-markers are the int8 greedy decode. Every marker lands on the float line — the int8 model picks the same token at every step (48/48). The sawtooth is the `+1 (mod 8)` counter each prompt continues from its last seed token.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/tiny_generate_int8)
