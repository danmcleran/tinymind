---
title: On-Target Benchmarks
layout: default
nav_order: 14
---

# On-Target Benchmarks: tinymind vs TFLM vs emlearn

tinymind, TensorFlow Lite for Microcontrollers (TFLM, now LiteRT for Microcontrollers) and emlearn were run on the same microcontrollers: a Cortex-M4F (NUCLEO-F411RE) and a [Cortex-M0+](#cortex-m0-nucleo-l031k6) (NUCLEO-L031K6). Every framework executed the same models, and the int8 frameworks used the same int8 weights. Everything was measured on the board; nothing was simulated.

{: .note }
F411 measured 2026-10-13, Cortex-M0+ measured 2026-10-17, with the code in [`benchmarks/framework_comparison_f411`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411). These are numbers from two boards, one compiler and three models. They describe this setup, not every workload. Updated after [#192](https://github.com/danmcleran/tinymind/pull/192), [#195](https://github.com/danmcleran/tinymind/pull/195), [#198](https://github.com/danmcleran/tinymind/pull/198), [#199](https://github.com/danmcleran/tinymind/pull/199), [#202](https://github.com/danmcleran/tinymind/pull/202), [#204](https://github.com/danmcleran/tinymind/pull/204), [#208](https://github.com/danmcleran/tinymind/pull/208), [#210](https://github.com/danmcleran/tinymind/pull/210) and [#211](https://github.com/danmcleran/tinymind/pull/211), nine tinymind fixes this benchmark prompted (see [what changed](#what-changed-since-the-first-run)).

## Setup

| | |
|---|---|
| Board | NUCLEO-F411RE: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on |
| Compiler | arm-none-eabi-gcc 13.2.1, `-mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16` |
| Timing | DWT cycle counter; median of 33 warm inferences (spread under 0.1%) |
| tinymind | master @ `b02e92d`, int8 layer family, with and without the `TINYMIND_ENABLE_SIMD_ARM_DSP` (SMLAD) backend |
| TFLM | tflite-micro @ `18811372` (2026-10-12), with CMSIS-NN kernels and with reference kernels |
| emlearn | 0.23.2 (`eml_net`, float32) |
| Converter | TensorFlow 2.21, post-training int8 quantization with default settings (per-channel weights) |

### Models

- **MLP 64 → 32 → 16 → 10** (ReLU), trained on the scikit-learn digits dataset and evaluated on 360 held-out samples. All three frameworks run it. The digits are scikit-learn's copy of the UCI [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49) dataset (Alpaydin and Kaynak, 1998), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Keyword-spotting-style CNN** with the same shape as [`examples/kws_cortex_m_int8`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m_int8): 20×20×1 input → conv 3×3×8 → max-pool 2×2 → depthwise 3×3 → pointwise 8→16 → average-pool → dense 10. It uses random weights, so it measures speed and exactness, not accuracy. tinymind and TFLM only; emlearn has no convolution.

### Keeping it fair

1. Each network starts as **one float model**. TFLite quantizes it to int8.
2. The int8 weights, int32 biases and requantization multipliers/shifts are copied **unchanged** out of the `.tflite` file into tinymind's `QConv2DPerChannel`, `QDepthwiseConv2D`, `QPointwiseConv2DPerChannel` and pooling layers. Both runtimes therefore execute the same integer arithmetic.
3. On the board, the int8 outputs of every test sample are folded into a hash. **tinymind and TFLM (CMSIS-NN and reference kernels) produce identical hashes for both models at every optimization level.** The results are bit-exact, so accuracy is identical by construction.
4. The optimization level is applied to every framework at once. tinymind and emlearn are compiled at `-O2` or `-O3`. TFLM is linked either against its default release build (kernels `-O2`, core `-Os`) or against a rebuild with every level forced to `-O3`.
5. **Code + weights** is the image size minus an empty-harness image, minus the embedded test data. **RAM** is `.data + .bss` plus peak stack, minus the same harness. The TFLM tensor arena is the smallest size `AllocateTensors()` accepts, found by bisecting on the board.
6. emlearn's final softmax is replaced with identity so that every backend returns logits. The predicted class is unchanged.

## Results

### MLP (digits, 360 test samples)

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| tinymind int8 (scalar) | 20,639 | 20,486 | 205 µs | 5.3–5.5 KB | 152 B | 96.11% |
| **tinymind int8 (+SMLAD)** | **10,305** | **10,241** | **102 µs** | 5.4–5.5 KB | 124 B | 96.11% |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 µs | 36.7–40.6 KB | 2,600 B | 96.11% |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 µs | 25.7–29.6 KB | 2,152 B | 96.11% |
| TFLM float32 | 27,902 | 28,059 | 281 µs | 43.2–47.3 KB | 1,824 B | 96.11% |
| emlearn float32 | 40,527 | 40,529 | 405 µs | 14.4–14.7 KB | 892 B | 96.11% |

### KWS-style CNN

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM |
|---|---:|---:|---:|---:|---:|
| tinymind int8 (scalar) | 266,451 | 254,900 | 2.55 ms | 5.0–6.8 KB | 5,332 B |
| **tinymind int8 (+SMLAD)** | **192,965** | **181,568** | **1.82 ms** | 4.7–6.4 KB | 5,296 B |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5.76 ms | 84.5–91.6 KB | 6,288 B |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24.4 ms | 41.1–46.2 KB | 5,668 B |
| TFLM float32 | 2,136,067 | 2,159,717 | 21.6 ms | 84.3–91.6 KB | 15,128 B |

Code + weights ranges are `-O2` / `-O3`.

TFLM also spends 133k–163k cycles (1.3–1.6 ms) in `AllocateTensors()` once at boot. tinymind has no setup step, because its layers are `const` aggregates initialized at compile time.

### Per-layer cycles, KWS CNN

Each layer was timed separately, behind its own non-inlined call.

| Layer | tinymind +SMLAD `-O2` | tinymind +SMLAD `-O3` | TFLM CMSIS-NN `-O2` | TFLM CMSIS-NN `-O3` |
|---|---:|---:|---:|---:|
| conv 3×3, 1→8 | 132,155 | **127,756** | 434,553 | 435,412 |
| max-pool 2×2 | 10,162 | **10,160** | 35,807 | 36,697 |
| depthwise 3×3 | 21,415 | **21,047** | 39,208 | 40,918 |
| pointwise 8→16 | **22,941** | 25,865 | 53,212 | 53,866 |
| average-pool | **2,267** | 2,281 | 8,875 | 7,917 |
| dense 16→10 | 948 | **946** | 2,097 | 2,041 |

## What the numbers say

**tinymind with the SMLAD backend is faster than TFLM with CMSIS-NN on every layer, at both optimization levels.** On the MLP it is 27% faster (10.2k vs 14.0k cycles at `-O3`). On the CNN it is 68% faster at `-O3` (182k vs 576k, 3.2× the speed) and 66% faster at `-O2`. Code + weights are 5–7 KB against 37–92 KB (7–18× smaller), because there is no interpreter or flatbuffer parser, and no unused kernel variants get linked in.

**Without SMLAD,** tinymind's portable scalar build still runs the CNN 56% faster than TFLM with CMSIS-NN (255k vs 576k cycles at `-O3`). On the MLP the scalar build (20.5k) sits between CMSIS-NN (14.0k) and TFLM's reference kernels (31.4k).

**Where the time goes now.** The single-channel 3×3 stem is still about two thirds of the CNN's time, at 29% of CMSIS-NN's cost for the same layer (128k vs 435k cycles), so it remains the layer to work on.

**Build tinymind at `-O3`.** The MLP runs within 1% at either level, and the CNN is 6% slower at `-O2` with SMLAD (193k vs 182k) and 5% slower without it. Individual layers can still go either way: pointwise is faster at `-O2` here. tinymind's kernels are templates compiled in your own translation unit, so the optimizer's choices are part of its performance; CMSIS-NN's kernels ship precompiled and barely move between levels.

**RAM is close on the CNN.** tinymind keeps every intermediate activation buffer static. TFLM's arena planner reuses buffers but carries interpreter state. Alternating between two buffers (2,592 B and 784 B instead of 4,432 B) would save about 1 KB.

**Among the float runtimes,** TFLM float32 is faster than emlearn on the MLP. emlearn has the smallest float footprint, and almost all of it is the weights themselves.

## Cortex-M0+: NUCLEO-L031K6

The same harness, backends and models on the smallest class of part, plus one model that emlearn can also run in integer arithmetic.

| | |
|---|---|
| Board | NUCLEO-L031K6: STM32L031K6, Cortex-M0+ (ARMv6-M) at 32 MHz, 1 flash wait state with prefetch on; 32 KB flash, 8 KB SRAM; no FPU, no DSP extension, no hardware divide |
| Compiler | arm-none-eabi-gcc 13.2.1, `-mcpu=cortex-m0plus -mthumb -mfloat-abi=soft` |
| Timing | The M0+ has no DWT cycle counter: SysTick runs at the core clock over its full 24-bit range and its interrupt counts wraps, giving a free-running 32-bit cycle count (checked on the board against a 3-cycle busy loop and for monotonicity across ~110k wraps) |
| Test sets | The first 32 MLP and random-forest samples (360 do not fit in 32 KB of flash next to the code); all 8 KWS samples |
| tinymind | master @ `576f146`, scalar int8 (ARMv6-M has no SMLAD) |

### MLP (digits, first 32 test samples)

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **tinymind int8** | **35,637** | **35,963** | **1.12 ms** | 5.5–5.8 KB | 200 B | 96.88% |
| TFLM int8 (CMSIS-NN) | does not fit | does not fit | - | 39.2–43.4 KB | 1,872+ B | - |
| TFLM int8 (reference) | does not fit | does not fit | - | 33.9–38.3 KB | 1,776+ B | - |
| TFLM float32 | does not fit | does not fit | - | 45.7–50.0 KB | 1,392+ B | - |
| emlearn float32 (software float) | 712,272 | 707,113 | 22.1 ms | 18.7–19.0 KB | 972 B | 96.88% |

### KWS-style CNN

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM |
|---|---:|---:|---:|---:|---:|
| **tinymind int8** | **818,262** | **746,112** | **23.3 ms** | 6.0–8.5 KB | 5,488 B |
| TFLM int8 (CMSIS-NN) | does not fit | does not fit | - | 83.0–90.5 KB | 5,136+ B |
| TFLM int8 (reference) | does not fit | does not fit | - | 51.1–57.1 KB | 5,072+ B |
| TFLM float32 | does not fit | does not fit | - | 82.9–90.3 KB | 14,448+ B |

"Does not fit" sizes come from linking the same images against an unbounded memory map; their RAM is static RAM, tensor arena included, without stack, so it is a floor.

### Random forest (10 trees, depth 6, digits, first 32 test samples)

emlearn has no int8 MLP, so the MLP row above is int8 against software float. Trees are the model both libraries run in pure integer arithmetic: one scikit-learn `RandomForestClassifier` with majority voting, on integer features, emitted for each. Each backend's predictions match its own host reference on the board.

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| tinymind int8, run-time walk (`QDecisionTree`) | 2,581 | 2,524 | 79 µs | 12.8–12.9 KB | 80 B | 96.88% |
| **tinymind int8, compile-time compact (`QStaticDecisionTree`, `QTreeCompactEval`)** | 1,705 | 1,649 | 52 µs | **3.0–3.1 KB** | **72 B** | 96.88% |
| tinymind int8, compile-time inline (`QStaticDecisionTree`, `QTreeInlineEval`) | 919 | 872 | 27 µs | 6.8–8.3 KB | 120 B | 96.88% |
| **tinymind uint8, compile-time inline (`QStaticDecisionTree`, `QTreeInlineEval`)** | **872** | 845 | 26 µs | 6.2–7.9 KB | 112 B | 96.88% |
| emlearn int16, inline (generated if/else) | 917 | **841** | 26 µs | 6.0–8.3 KB | 184 B | 96.88% |
| emlearn int16, loadable | 6,267 | 6,409 | 200 µs | 7.1–7.1 KB | 336 B | 96.88% |

A tree walk's length depends on the input, so the spread between fastest and slowest inference is 7–10% here (26% for emlearn's `loadable`), against under 0.1% for the networks.

### What the M0+ numbers say

**Only tinymind fits the networks.** TFLM's runtime plus the MLP alone needs 34–50 KB, more than the part's 32 KB of flash before the harness or a test input is added. tinymind runs the MLP in 5.5 KB and the CNN in 6.0 KB at `-O2`, bit-exact with the host and the F411 on a core that has no unaligned access.

**On the MLP, int8 against software float is 20×** (35.6k vs 712k cycles). That is not int8 against int8, since emlearn has no int8 MLP; it is what each library offers for this network on a part without an FPU.

**The random forest went from a 3× loss to a tie.** The first run walked the forest with `QDecisionTree`: about 20 Thumb-1 instructions per node and a full 12-byte node per leaf, against emlearn's generated load, compare-immediate and branch. [`QStaticDecisionTree`]({{ site.baseurl }}/architectures/decision-trees#compile-time-models) (#223) generates that same code from the same node pool at compile time, so the pool is not stored at all. It is 2.9× faster than the walk in about half its size. With `uint8_t` features, which load with Thumb-1's immediate-offset `LDRB` instead of a register-offset `LDRSB`, it beats emlearn at `-O2` (872 vs 917 cycles), is within 0.5% at `-O3` (845 vs 841), is smaller at `-O3` (7.9 vs 8.3 KB) and uses less RAM (112 vs 184 B). When flash matters more than cycles, the compact policy packs the same pool at compile time into 4-byte decision nodes in pre-order and a deduplicated int8 leaf table: the whole forest is 3.0 KB, half of emlearn's smallest and a quarter of the walk, at 1.5× the walk's speed and about half of inline's. Against emlearn's data-driven `loadable` method it is 3.7–3.9× faster in 42% of the flash. The model is also checked when it compiles: a split on a feature the input does not have, or a child index outside the pool, is a build error instead of an out-of-bounds read.

**M0+ against M4F, same code.** The MLP takes 1.8× the cycles of the F411's scalar build and the CNN 2.9×. Requantization's 64-bit multiply becomes a library call (`__aeabi_lmul`) on ARMv6-M, and the 3×3 stem is 69% of the CNN, so those are where M0+ tuning would start.

## What changed since the first run

The first runs (tinymind before #192) exposed four weak spots. All four are fixed, along with five follow-ups. Every int8 output hash is unchanged, so every fix is bit-exact.

| tinymind | First run | Now |
|---|---:|---:|
| MLP +SMLAD `-O2` | 24,591 | 10,305 |
| MLP +SMLAD `-O3` | 12,278 | 10,241 |
| MLP scalar `-O3` | 33,154 | 20,486 |
| CNN +SMLAD `-O2` | 831,570 | 192,965 |
| CNN +SMLAD `-O3` | 422,449 | 181,568 |
| CNN scalar `-O3` | 445,579 | 254,900 |

- **[#192](https://github.com/danmcleran/tinymind/pull/192), SMLAD inner loop.** At `-O2` the int8 dot product stayed rolled inside each layer's output loop, where the requantizer's hoisted constants left it about three scratch registers. In the disassembly it issued 2 SMLADs per iteration, stored each loaded word to the stack and read it back, and reloaded its loop bound from memory on every pass. How badly depended on unrelated surrounding code: with identical kernels the MLP measured anywhere from 14.5k to 24.6k cycles. The loop is now unrolled with a pragma (so `-O2` compiles it like `-O3`), and the input zero point is folded into `SXTAB16`, the way CMSIS-NN does it.
- **[#195](https://github.com/danmcleran/tinymind/pull/195), convolution windows.** The conv layers called the dot product once per kernel tap, so the single-channel 3×3 stem ran nine length-1 dot products per output, none of them long enough to use SMLAD. Each output pixel's kernel window is now gathered into a small contiguous patch once (up to 64 bytes; larger windows run one dot product per kernel row instead) and reused by every filter.
- **[#198](https://github.com/danmcleran/tinymind/pull/198), depthwise and pointwise loops.** Both computed one output at a time, so every output re-fetched its filter's weights and requantizer. Pointwise now runs filter-outer, and depthwise runs channel-outer with the input zero point folded into the bias once per channel (`Σw·(x−zp) = Σw·x − zp·Σw`). Depthwise went from 58k to 31k cycles and pointwise from 67k to 44k, both now ahead of CMSIS-NN.
- **[#199](https://github.com/danmcleran/tinymind/pull/199), scalar dot product.** The portable scalar dot product got the same unroll hint as the SMLAD loop: the scalar MLP dropped from 31k–33k to 21.5k cycles.
- **[#202](https://github.com/danmcleran/tinymind/pull/202), convolution tiles.** After #195 the stem still reloaded every filter's weights, bias and requantizer at every output pixel. The patch path now gathers a tile of windows (up to `TINYMIND_CONV_TILE_MAX_BYTES`, 256 bytes by default, which is a whole output row for this stem) and sweeps it one filter at a time. The stem dropped from 260k to 177k cycles at `-O3`, for about 0.2 KB more peak stack.
- **[#204](https://github.com/danmcleran/tinymind/pull/204), pooling loops.** At `-O2` the pooling window loops stayed rolled. With the same unroll hints as the conv kernels, max-pool went from 14.5k to 10.2k cycles and the average pool from 7.2k to 2.3k, level with `-O3`.
- **[#208](https://github.com/danmcleran/tinymind/pull/208), requantization.** Every int8 output rounded twice, gemmlowp style: a rounding high multiply, then a rounding shift with a mask, threshold and compare. In the stem that was about 30 instructions per output, with seven constants spilled to the stack. For a multiplier ≥ 0 and a right shift of at least 2, which covers every layer here, the two roundings combine exactly into one 64-bit rounding: an `SMLAL` and an `ASR` on the M4, with a bit-identical result. The stem dropped from 177k to 158k cycles at `-O3`, depthwise from 31k to 23k and pointwise from 44k to 39k. The MLP dropped from 11.2k to 10.2k.
- **[#210](https://github.com/danmcleran/tinymind/pull/210), stem registers.** The stem's output loop still reloaded state on every output. An int8 output store may alias the int8 weight array, so the compiler re-read the filter's weights after each store; the scalar build re-read all nine. And the requantizer kept six values live, which spilled to the stack. The filter's weights are now copied into a local array, which a store cannot alias, and the requantizer is decoded once per filter: the zero point goes into the rounding constant, and the clamp to the int8 range is a single `SSAT`. Layers with a fused activation clamp keep the general path. The stem dropped from 158k to 128k cycles at `-O3` and from 190k to 132k at `-O2`.
- **[#211](https://github.com/danmcleran/tinymind/pull/211), pointwise and depthwise registers.** The same two fixes applied to the other conv kernels. Pointwise dropped from 39k to 26k cycles at `-O3` (41k to 23k at `-O2`) and depthwise from 23k to 21k.

## Integration notes

- **emlearn:** `pip install emlearn`, convert a scikit-learn model, include one header. Integration takes minutes. It covers MLPs and tree ensembles in float.
- **TFLM:** clone and build with `make` (the build downloads CMSIS, CMSIS-NN and flatbuffers). Model conversion needs a full TensorFlow install. Details that cost time in this exercise:
  - `arena_used_bytes()` reported 100–300 B less than the minimum arena `AllocateTensors()` accepts.
  - The per-operator profiler is compiled out of release builds, so a separate `release_with_logs` library is needed to profile.
  - A Keras `Flatten` with a dynamic batch dimension adds shape-computation operators unless the model is built with `batch_size=1`.
- **tinymind:** header-only, with nothing to build or link. It has no `.tflite` importer. The bridge that copied the TFLite int8 parameters into tinymind layers was written for this benchmark. tinymind ships [PyTorch, ONNX and tree-ensemble importers](https://github.com/danmcleran/tinymind/tree/master/apps); the tree importer covers scikit-learn, XGBoost and LightGBM.

## Reproduce it

Everything is in [`benchmarks/framework_comparison_f411`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411): bare-metal board support, a timing harness, one backend per framework, the model generator, and the raw results behind these tables. With `arm-none-eabi-gcc`, `openocd` and a NUCLEO-F411RE attached:

```
cd benchmarks/framework_comparison_f411
make deps            # Python venv + TFLM clone at the pinned revision + TFLM libraries
make host-check      # tinymind int8 outputs reproduce the expected hashes
make LEVEL=O3 run    # build, flash and capture every image
```

On a NUCLEO-L031K6, `make BOARD=l031 LEVEL=O3 run` builds and runs the Cortex-M0+ images (`results/raw_l031_O3.txt`), and `make BOARD=l031 fit` reports what the images that do not fit would need.

The generated model headers are committed, so the firmware builds without TensorFlow; `make deps` is only needed for the TFLM and emlearn images.

## A note on host references

TensorFlow's pip `tf.lite.Interpreter` differs from all three on-device int8 paths by ±1 LSB on 35 of 3,600 MLP outputs: its fully-connected kernel rounds once where TFLM, CMSIS-NN and tinymind round twice. If you validate an MCU int8 pipeline against the desktop interpreter, expect occasional last-bit differences that are not bugs.
