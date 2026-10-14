---
title: On-Target Benchmarks
layout: default
nav_order: 14
---

# On-Target Benchmarks: TinyMind vs TFLM vs emlearn

TinyMind, TensorFlow Lite for Microcontrollers (TFLM, now LiteRT for Microcontrollers) and emlearn were run on the same microcontroller. Every framework executed the same two models, and the int8 frameworks used the same int8 weights. Everything was measured on the board; nothing was simulated.

{: .note }
Measured 2026-10-13 with the code in [`benchmarks/framework_comparison_f411`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411). These are numbers from one board, one compiler and two models. They describe this setup, not every workload. Updated after [#192](https://github.com/danmcleran/tinymind/pull/192), [#195](https://github.com/danmcleran/tinymind/pull/195), [#198](https://github.com/danmcleran/tinymind/pull/198), [#199](https://github.com/danmcleran/tinymind/pull/199), [#202](https://github.com/danmcleran/tinymind/pull/202), [#204](https://github.com/danmcleran/tinymind/pull/204) and [#208](https://github.com/danmcleran/tinymind/pull/208), seven TinyMind fixes this benchmark prompted (see [what changed](#what-changed-since-the-first-run)).

## Setup

| | |
|---|---|
| Board | NUCLEO-F411RE: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on |
| Compiler | arm-none-eabi-gcc 13.2.1, `-mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16` |
| Timing | DWT cycle counter; median of 33 warm inferences (spread under 0.1%) |
| TinyMind | master @ `0a4db84`, int8 layer family, with and without the `TINYMIND_ENABLE_SIMD_ARM_DSP` (SMLAD) backend |
| TFLM | tflite-micro @ `18811372` (2026-10-12), with CMSIS-NN kernels and with reference kernels |
| emlearn | 0.23.2 (`eml_net`, float32) |
| Converter | TensorFlow 2.21, post-training int8 quantization with default settings (per-channel weights) |

### Models

- **MLP 64 → 32 → 16 → 10** (ReLU), trained on the scikit-learn digits dataset and evaluated on 360 held-out samples. All three frameworks run it.
- **Keyword-spotting-style CNN** with the same shape as [`examples/kws_cortex_m_int8`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m_int8): 20×20×1 input → conv 3×3×8 → max-pool 2×2 → depthwise 3×3 → pointwise 8→16 → average-pool → dense 10. It uses random weights, so it measures speed and exactness, not accuracy. TinyMind and TFLM only; emlearn has no convolution.

### Keeping it fair

1. Each network starts as **one float model**. TFLite quantizes it to int8.
2. The int8 weights, int32 biases and requantization multipliers/shifts are copied **unchanged** out of the `.tflite` file into TinyMind's `QConv2DPerChannel`, `QDepthwiseConv2D`, `QPointwiseConv2DPerChannel` and pooling layers. Both runtimes therefore execute the same integer arithmetic.
3. On the board, the int8 outputs of every test sample are folded into a hash. **TinyMind and TFLM (CMSIS-NN and reference kernels) produce identical hashes for both models at every optimization level.** The results are bit-exact, so accuracy is identical by construction.
4. The optimization level is applied to every framework at once. TinyMind and emlearn are compiled at `-O2` or `-O3`. TFLM is linked either against its default release build (kernels `-O2`, core `-Os`) or against a rebuild with every level forced to `-O3`.
5. **Code + weights** is the image size minus an empty-harness image, minus the embedded test data. **RAM** is `.data + .bss` plus peak stack, minus the same harness. The TFLM tensor arena is the smallest size `AllocateTensors()` accepts, found by bisecting on the board.
6. emlearn's final softmax is replaced with identity so that every backend returns logits. The predicted class is unchanged.

## Results

### MLP (digits, 360 test samples)

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| TinyMind int8 (scalar) | 20,639 | 20,486 | 205 µs | 5.3–5.5 KB | 152 B | 96.11% |
| **TinyMind int8 (+SMLAD)** | **10,305** | **10,241** | **102 µs** | 5.4–5.5 KB | 124 B | 96.11% |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 µs | 36.7–40.6 KB | 2,600 B | 96.11% |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 µs | 25.7–29.6 KB | 2,152 B | 96.11% |
| TFLM float32 | 27,902 | 28,059 | 281 µs | 43.2–47.3 KB | 1,824 B | 96.11% |
| emlearn float32 | 40,527 | 40,529 | 405 µs | 14.4–14.7 KB | 892 B | 96.11% |

### KWS-style CNN

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM |
|---|---:|---:|---:|---:|---:|
| TinyMind int8 (scalar) | 329,733 | 272,639 | 2.73 ms | 4.1–5.7 KB | 5,164 B |
| **TinyMind int8 (+SMLAD)** | **276,387** | **229,438** | **2.29 ms** | 3.9–5.8 KB | 5,164 B |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5.76 ms | 84.5–91.6 KB | 6,288 B |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24.4 ms | 41.1–46.2 KB | 5,668 B |
| TFLM float32 | 2,136,067 | 2,159,717 | 21.6 ms | 84.3–91.6 KB | 15,128 B |

Code + weights ranges are `-O2` / `-O3`.

TFLM also spends 133k–163k cycles (1.3–1.6 ms) in `AllocateTensors()` once at boot. TinyMind has no setup step, because its layers are `const` aggregates initialized at compile time.

### Per-layer cycles, KWS CNN

Each layer was timed separately, behind its own non-inlined call.

| Layer | TinyMind +SMLAD `-O2` | TinyMind +SMLAD `-O3` | TFLM CMSIS-NN `-O2` | TFLM CMSIS-NN `-O3` |
|---|---:|---:|---:|---:|
| conv 3×3, 1→8 | 189,920 | **157,571** | 434,553 | 435,412 |
| max-pool 2×2 | 10,162 | **10,154** | 35,807 | 36,697 |
| depthwise 3×3 | 26,044 | **23,322** | 39,208 | 40,918 |
| pointwise 8→16 | 41,121 | **38,659** | 53,212 | 53,866 |
| average-pool | **2,269** | 2,281 | 8,875 | 7,917 |
| dense 16→10 | **942** | 956 | 2,097 | 2,041 |

## What the numbers say

**TinyMind with the SMLAD backend is faster than TFLM with CMSIS-NN on every layer, at both optimization levels.** On the MLP it is 27% faster (10.2k vs 14.0k cycles at `-O3`). On the CNN it is 60% faster at `-O3` (229k vs 576k, 2.5× the speed) and 52% faster at `-O2`. Code + weights are 4–6 KB against 37–92 KB (7–22× smaller), because there is no interpreter or flatbuffer parser, and no unused kernel variants get linked in.

**Without SMLAD,** TinyMind's portable scalar build still runs the CNN 53% faster than TFLM with CMSIS-NN (273k vs 576k cycles at `-O3`). On the MLP the scalar build (20.5k) sits between CMSIS-NN (14.0k) and TFLM's reference kernels (31.4k).

**Where the time goes now.** The single-channel 3×3 stem is still about two thirds of the CNN's time, at 36% of CMSIS-NN's cost for the same layer (158k vs 435k cycles), so it remains the layer to work on.

**Build TinyMind at `-O3`.** The MLP runs within 1% at either level, but the CNN with SMLAD is 20% slower at `-O2` (276k vs 229k), most of it in the stem convolution (190k vs 158k). TinyMind's kernels are templates compiled in your own translation unit, so the optimizer's choices are part of its performance; CMSIS-NN's kernels ship precompiled and barely move between levels.

**RAM is close on the CNN.** TinyMind keeps every intermediate activation buffer static. TFLM's arena planner reuses buffers but carries interpreter state. Alternating between two buffers (2,592 B and 784 B instead of 4,432 B) would save about 1 KB.

**Among the float runtimes,** TFLM float32 is faster than emlearn on the MLP. emlearn has the smallest float footprint, and almost all of it is the weights themselves.

## What changed since the first run

The first runs (TinyMind before #192) exposed four weak spots. All four are fixed, along with three follow-ups. Every int8 output hash is unchanged, so every fix is bit-exact.

| TinyMind | First run | Now |
|---|---:|---:|
| MLP +SMLAD `-O2` | 24,591 | 10,305 |
| MLP +SMLAD `-O3` | 12,278 | 10,241 |
| MLP scalar `-O3` | 33,154 | 20,486 |
| CNN +SMLAD `-O2` | 831,570 | 276,387 |
| CNN +SMLAD `-O3` | 422,449 | 229,438 |
| CNN scalar `-O3` | 445,579 | 272,639 |

- **[#192](https://github.com/danmcleran/tinymind/pull/192), SMLAD inner loop.** At `-O2` the int8 dot product stayed rolled inside each layer's output loop, where the requantizer's hoisted constants left it about three scratch registers. In the disassembly it issued 2 SMLADs per iteration, stored each loaded word to the stack and read it back, and reloaded its loop bound from memory on every pass. How badly depended on unrelated surrounding code: with identical kernels the MLP measured anywhere from 14.5k to 24.6k cycles. The loop is now unrolled with a pragma (so `-O2` compiles it like `-O3`), and the input zero point is folded into `SXTAB16`, the way CMSIS-NN does it.
- **[#195](https://github.com/danmcleran/tinymind/pull/195), convolution windows.** The conv layers called the dot product once per kernel tap, so the single-channel 3×3 stem ran nine length-1 dot products per output, none of them long enough to use SMLAD. Each output pixel's kernel window is now gathered into a small contiguous patch once (up to 64 bytes; larger windows run one dot product per kernel row instead) and reused by every filter.
- **[#198](https://github.com/danmcleran/tinymind/pull/198), depthwise and pointwise loops.** Both computed one output at a time, so every output re-fetched its filter's weights and requantizer. Pointwise now runs filter-outer, and depthwise runs channel-outer with the input zero point folded into the bias once per channel (`Σw·(x−zp) = Σw·x − zp·Σw`). Depthwise went from 58k to 31k cycles and pointwise from 67k to 44k, both now ahead of CMSIS-NN.
- **[#199](https://github.com/danmcleran/tinymind/pull/199), scalar dot product.** The portable scalar dot product got the same unroll hint as the SMLAD loop: the scalar MLP dropped from 31k–33k to 21.5k cycles.
- **[#202](https://github.com/danmcleran/tinymind/pull/202), convolution tiles.** After #195 the stem still reloaded every filter's weights, bias and requantizer at every output pixel. The patch path now gathers a tile of windows (up to `TINYMIND_CONV_TILE_MAX_BYTES`, 256 bytes by default, which is a whole output row for this stem) and sweeps it one filter at a time. The stem dropped from 260k to 177k cycles at `-O3`, for about 0.2 KB more peak stack.
- **[#204](https://github.com/danmcleran/tinymind/pull/204), pooling loops.** At `-O2` the pooling window loops stayed rolled. With the same unroll hints as the conv kernels, max-pool went from 14.5k to 10.2k cycles and the average pool from 7.2k to 2.3k, level with `-O3`.
- **[#208](https://github.com/danmcleran/tinymind/pull/208), requantization.** Every int8 output rounded twice, gemmlowp style: a rounding high multiply, then a rounding shift with a mask, threshold and compare. In the stem that was about 30 instructions per output, with seven constants spilled to the stack. For a multiplier ≥ 0 and a right shift of at least 2, which covers every layer here, the two roundings combine exactly into one 64-bit rounding: an `SMLAL` and an `ASR` on the M4, with a bit-identical result. The stem dropped from 177k to 158k cycles at `-O3`, depthwise from 31k to 23k and pointwise from 44k to 39k. The MLP dropped from 11.2k to 10.2k.

## Integration notes

- **emlearn:** `pip install emlearn`, convert a scikit-learn model, include one header. Integration takes minutes. It covers MLPs and tree ensembles in float.
- **TFLM:** clone and build with `make` (the build downloads CMSIS, CMSIS-NN and flatbuffers). Model conversion needs a full TensorFlow install. Details that cost time in this exercise:
  - `arena_used_bytes()` reported 100–300 B less than the minimum arena `AllocateTensors()` accepts.
  - The per-operator profiler is compiled out of release builds, so a separate `release_with_logs` library is needed to profile.
  - A Keras `Flatten` with a dynamic batch dimension adds shape-computation operators unless the model is built with `batch_size=1`.
- **TinyMind:** header-only, with nothing to build or link. It has no `.tflite` importer. The bridge that copied the TFLite int8 parameters into TinyMind layers was written for this benchmark. TinyMind ships [PyTorch and ONNX importers](https://github.com/danmcleran/tinymind/tree/master/apps).

## Reproduce it

Everything is in [`benchmarks/framework_comparison_f411`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411): bare-metal board support, a timing harness, one backend per framework, the model generator, and the raw results behind these tables. With `arm-none-eabi-gcc`, `openocd` and a NUCLEO-F411RE attached:

```
cd benchmarks/framework_comparison_f411
make deps            # Python venv + TFLM clone at the pinned revision + TFLM libraries
make host-check      # TinyMind int8 outputs reproduce the expected hashes
make LEVEL=O3 run    # build, flash and capture every image
```

The generated model headers are committed, so the firmware builds without TensorFlow; `make deps` is only needed for the TFLM and emlearn images.

## A note on host references

TensorFlow's pip `tf.lite.Interpreter` differs from all three on-device int8 paths by ±1 LSB on 35 of 3,600 MLP outputs: its fully-connected kernel rounds once where TFLM, CMSIS-NN and TinyMind round twice. If you validate an MCU int8 pipeline against the desktop interpreter, expect occasional last-bit differences that are not bugs.
