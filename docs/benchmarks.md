---
title: On-Target Benchmarks
layout: default
nav_order: 14
---

# On-Target Benchmarks: tinymind vs TFLM vs emlearn

tinymind, TensorFlow Lite for Microcontrollers (TFLM, now LiteRT for Microcontrollers) and emlearn were run on the same microcontrollers: a Cortex-M4F (NUCLEO-F411RE) and a [Cortex-M0+](#cortex-m0-nucleo-l031k6) (NUCLEO-L031K6). Every framework executed the same models, and the int8 frameworks used the same int8 weights. Everything was measured on the board; nothing was simulated.

{: .note }
Every table re-measured 2026-11-12 on both boards, at tinymind `6a8b522` with [#275](https://github.com/danmcleran/tinymind/pull/275) (which changes only Cortex-M0+ code), with the code in [`benchmarks/framework_comparison_f411`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411). These are numbers from two boards, one compiler and five models. They describe this setup, not every workload. Updated after [#192](https://github.com/danmcleran/tinymind/pull/192), [#195](https://github.com/danmcleran/tinymind/pull/195), [#198](https://github.com/danmcleran/tinymind/pull/198), [#199](https://github.com/danmcleran/tinymind/pull/199), [#202](https://github.com/danmcleran/tinymind/pull/202), [#204](https://github.com/danmcleran/tinymind/pull/204), [#208](https://github.com/danmcleran/tinymind/pull/208), [#210](https://github.com/danmcleran/tinymind/pull/210) and [#211](https://github.com/danmcleran/tinymind/pull/211), nine tinymind fixes this benchmark prompted (see [what changed](#what-changed-since-the-first-run)), after [#229](https://github.com/danmcleran/tinymind/pull/229) and [#273](https://github.com/danmcleran/tinymind/pull/273), which the MLPerf Tiny models prompted, and after [#275](https://github.com/danmcleran/tinymind/pull/275), a Cortex-M0+ fix this refresh prompted.

## Setup

| | |
|---|---|
| Board | NUCLEO-F411RE: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on |
| Compiler | arm-none-eabi-gcc 13.2.1, `-mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16` (Clang 18.1.3 [below](#tinymind-under-clang)) |
| Timing | DWT cycle counter; median of 33 warm inferences (spread under 0.1%) |
| tinymind | master @ `6a8b522`, int8 layer family, with and without the `TINYMIND_ENABLE_SIMD_ARM_DSP` (SMLAD) backend |
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
| tinymind int8 (scalar) | 20,621 | 21,002 | 210 µs | 5.8 KB | 152 B | 96.11% |
| **tinymind int8 (+SMLAD)** | **10,167** | **10,227** | **102 µs** | 5.9 KB | 124 B | 96.11% |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 µs | 36.7–40.6 KB | 2,600 B | 96.11% |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 µs | 25.7–29.6 KB | 2,152 B | 96.11% |
| TFLM float32 | 27,902 | 28,059 | 281 µs | 43.2–47.3 KB | 1,824 B | 96.11% |
| emlearn float32 | 40,527 | 40,529 | 405 µs | 14.4–14.7 KB | 892 B | 96.11% |

### KWS-style CNN

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM |
|---|---:|---:|---:|---:|---:|
| tinymind int8 (scalar) | 271,043 | 254,766 | 2.55 ms | 5.8–6.6 KB | 5,308 B |
| **tinymind int8 (+SMLAD)** | **195,841** | **181,853** | **1.82 ms** | 5.4–6.3 KB | 5,372 B |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5.76 ms | 84.5–91.6 KB | 6,288 B |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24.4 ms | 41.1–46.2 KB | 5,668 B |
| TFLM float32 | 2,136,067 | 2,159,717 | 21.6 ms | 84.3–91.6 KB | 15,128 B |

Code + weights ranges are `-O2` / `-O3`.

TFLM also spends 133k–163k cycles (1.3–1.6 ms) in `AllocateTensors()` once at boot. tinymind has no setup step, because its layers are `const` aggregates initialized at compile time.

### Per-layer cycles, KWS CNN

Each layer was timed separately, behind its own non-inlined call.

| Layer | tinymind +SMLAD `-O2` | tinymind +SMLAD `-O3` | TFLM CMSIS-NN `-O2` | TFLM CMSIS-NN `-O3` |
|---|---:|---:|---:|---:|
| conv 3×3, 1→8 | 134,793 | **126,262** | 434,553 | 435,387 |
| max-pool 2×2 | 10,161 | **10,156** | 35,807 | 36,383 |
| depthwise 3×3 | 20,670 | **20,660** | 39,208 | 40,909 |
| pointwise 8→16 | 29,877 | **27,485** | 53,212 | 53,863 |
| average-pool | **2,269** | 2,281 | 8,875 | 7,921 |
| dense 16→10 | 954 | **952** | 2,097 | 2,031 |

## What the numbers say

**tinymind with the SMLAD backend is faster than TFLM with CMSIS-NN on every layer, at both optimization levels.** On the MLP it is 27% faster (10.2k vs 14.0k cycles at `-O3`). On the CNN it is 68% faster at `-O3` (182k vs 576k, 3.2× the speed) and 66% faster at `-O2`. Code + weights are 5–7 KB against 37–92 KB (6–17× smaller), because there is no interpreter or flatbuffer parser, and no unused kernel variants get linked in.

**Without SMLAD,** tinymind's portable scalar build still runs the CNN 56% faster than TFLM with CMSIS-NN (255k vs 576k cycles at `-O3`). On the MLP the scalar build (21.0k) sits between CMSIS-NN (14.0k) and TFLM's reference kernels (31.4k).

**Where the time goes now.** The single-channel 3×3 stem is still about two thirds of the CNN's time, at 29% of CMSIS-NN's cost for the same layer (126k vs 435k cycles), so it remains the layer to work on.

**Build tinymind at `-O3`.** The MLP runs within 2% at either level, and the CNN is 8% slower at `-O2` with SMLAD (196k vs 182k) and 6% slower without it. tinymind's kernels are templates compiled in your own translation unit, so the optimizer's choices are part of its performance; CMSIS-NN's kernels ship precompiled and barely move between levels.

**RAM is close on the CNN.** tinymind keeps every intermediate activation buffer static. TFLM's arena planner reuses buffers but carries interpreter state. Alternating between two buffers (2,592 B and 784 B instead of 4,432 B) would save about 1 KB.

**Among the float runtimes,** TFLM float32 is faster than emlearn on the MLP. emlearn has the smallest float footprint, and almost all of it is the weights themselves.

## MLPerf® Tiny reference models

The models above were built for this benchmark. To check the comparison on standard models, the F411 also runs two of the [MLPerf Tiny](https://github.com/mlcommons/tiny) reference int8 models, unmodified, with tinymind and with TFLM: keyword spotting (DS-CNN, 49×10 MFCC input, 12 classes) and image classification (ResNet-8, 32×32×3 input, 10 classes).

{: .note }
**These are not MLPerf results.** They were not produced with the MLPerf Tiny benchmark runner or under its rules: there is no accuracy run on the benchmark datasets and no energy measurement, and MLCommons has not verified them. They compare two inference stacks on the same reference models and the same inputs, and should not be compared with published MLPerf Tiny scores. MLPerf® is a registered trademark of MLCommons Association in the United States and other countries; see [mlcommons.org](https://mlcommons.org).

How they were run:

- **Same arithmetic.** A generator reads each `.tflite` and rebuilds the graph from tinymind layers, using the model's own int8 weights and int32 biases and the requantizer multipliers TFLM derives from its scales. Every output of every image matched between tinymind and TFLM, with CMSIS-NN and with reference kernels, at `-O2` and `-O3`.
- **Inputs.** 8 seeded random int8 tensors per model, not dataset samples. This measures latency, memory and agreement; accuracy is a property of the reference models, and it is MLCommons' to report.
- **Not redistributed.** The models are Apache-2.0 (MLCommons). The repository has the generator, not the models: `make mlperf-models` downloads them at a pinned commit and checks their SHA-256.

`-O3`, 100 MHz. RAM is static RAM plus peak stack, and includes TFLM's tensor arena. Flash includes the weights.

| Model | Framework | Cycles | ms | Flash | RAM |
|---|---|---:|---:|---:|---:|
| Keyword spotting (DS-CNN) | **tinymind +SMLAD** | **8,011,750** | **80.1** | **53,452** | **17,276** |
| | tinymind scalar | 15,677,563 | 156.8 | 53,072 | 17,288 |
| | TFLM CMSIS-NN | 9,583,099 | 95.8 | 148,864 | 24,432 |
| | TFLM reference | 67,484,238 | 674.8 | 101,944 | 23,828 |
| Image classification (ResNet-8) | **tinymind +SMLAD** | **26,368,775** | **263.7** | **112,352** | **54,416** |
| | tinymind scalar | 85,326,662 | 853.3 | 124,140 | 54,368 |
| | TFLM CMSIS-NN | 37,372,049 | 373.7 | 191,212 | 56,608 |
| | TFLM reference | 191,555,451 | 1,915.6 | 155,412 | 55,396 |

Per-layer cycles for the image classification model, `-O3` (each layer timed separately):

| Layer | tinymind +SMLAD | TFLM CMSIS-NN |
|---|---:|---:|
| conv 3×3, 3→16 | 1,920,193 | 2,837,416 |
| conv 3×3, 16→16 (×2) | 5,020,704 | 6,984,937 |
| conv 3×3 stride 2, 16→32 | 2,545,177 | 3,171,053 |
| conv 3×3, 32→32 | 4,414,753 | 5,718,998 |
| conv 1×1 stride 2, 16→32 | 490,899 | 677,296 |
| conv 3×3 stride 2, 32→64 | 2,239,423 | 2,747,445 |
| conv 3×3, 64→64 | 4,222,239 | 5,213,962 |
| conv 1×1 stride 2, 32→64 | 450,110 | 533,502 |
| residual add (32×32×16) | 442,425 | 1,407,630 |
| residual add (16×16×32) | 213,055 | 702,260 |
| residual add (8×8×64) | 102,458 | 350,923 |

What these numbers say:

- **tinymind takes 16.4% fewer cycles than CMSIS-NN on keyword spotting and 29.4% fewer on image classification**, and fewer on every layer of both models at `-O2` and `-O3`. It needs 36% of TFLM's flash on keyword spotting and 59% on image classification, and less RAM on both (17.3 against 24.4 KB, and 54.4 against 56.6 KB).
- **The first run lost image classification by 16%.** Before this suite, tinymind's 3×3 convolutions over 16 to 64 channels computed one output at a time, and SAME padding needed a separate pass and a padded copy of each input. Three changes closed it, all bit-exact:
  - The convolutions now compute two filters by two output pixels per pass through an SMLAD kernel, so each load feeds two multiplies, as CMSIS-NN does. The kernel is a separate function on purpose: inlined into the layer, it competed with the surrounding loops for registers, and these layers' cost swung by 30% with unrelated code changes. The 1×1 convolutions over 32 or more channels use the same kernel.
  - Padding is folded into the convolution and depthwise layers (`PadTop` … `PadRight` template parameters). A ring of padded rows on the stack replaces the 12 KB and 18 KB padded copies that had put tinymind's RAM above TFLM's.
  - A new `QSoftmaxFixedPoint1D` reproduces the TFLite int8 softmax exactly, so a model that ends in a softmax can be checked against TFLM bit for bit.
- **The residual adds take a third of CMSIS-NN's cycles** since [#273](https://github.com/danmcleran/tinymind/pull/273). `QAdd` rescaled each input with a two-step rounding and a branch on the shift's sign; for TFLite ADD without a fused activation, each input is now one 64-bit rounding (an `SMLAL` and a shift), proved equal to the reference by CBMC, and the output uses the same single-rounding requantizer as the convolutions. The three adds dropped from 557k, 336k and 176k cycles to 442k, 213k and 102k, bit-exact.

Reproduce with `make mlperf-models`, then `make SUITE=mlperf LEVEL=O3 run`; full tables, `-O2` included, are in [`results/REPORT_mlperf.md`](https://github.com/danmcleran/tinymind/blob/master/benchmarks/framework_comparison_f411/results/REPORT_mlperf.md).

## Cortex-M0+: NUCLEO-L031K6

The same harness, backends and models on the smallest class of part, plus one model that emlearn can also run in integer arithmetic.

| | |
|---|---|
| Board | NUCLEO-L031K6: STM32L031K6, Cortex-M0+ (ARMv6-M) at 32 MHz, 1 flash wait state with prefetch on; 32 KB flash, 8 KB SRAM; no FPU, no DSP extension, no hardware divide |
| Compiler | arm-none-eabi-gcc 13.2.1, `-mcpu=cortex-m0plus -mthumb -mfloat-abi=soft` |
| Timing | The M0+ has no DWT cycle counter: SysTick runs at the core clock over its full 24-bit range and its interrupt counts wraps, giving a free-running 32-bit cycle count (checked on the board against a 3-cycle busy loop and for monotonicity across ~110k wraps) |
| Test sets | The first 32 MLP and random-forest samples (360 do not fit in 32 KB of flash next to the code); all 8 KWS samples |
| tinymind | master @ `6a8b522`, scalar int8 (ARMv6-M has no SMLAD) |

### MLP (digits, first 32 test samples)

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **tinymind int8** | **33,010** | **32,644** | **1.02 ms** | 6.1–7.6 KB | 368 B | 96.88% |
| TFLM int8 (CMSIS-NN) | does not fit | does not fit | - | 39.2–43.4 KB | 1,872+ B | - |
| TFLM int8 (reference) | does not fit | does not fit | - | 33.9–38.3 KB | 1,776+ B | - |
| TFLM float32 | does not fit | does not fit | - | 45.7–50.0 KB | 1,392+ B | - |
| emlearn float32 (software float) | 712,272 | 707,113 | 22.1 ms | 18.7–19.0 KB | 972 B | 96.88% |

### KWS-style CNN

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM |
|---|---:|---:|---:|---:|---:|
| **tinymind int8** | **821,251** | **740,279** | **23.1 ms** | 7.7–8.2 KB | 5,596 B |
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

**Only tinymind fits the networks.** TFLM's runtime plus the MLP alone needs 34–50 KB, more than the part's 32 KB of flash before the harness or a test input is added. tinymind runs the MLP in 7.6 KB and the CNN in 7.7 KB at `-O2`, bit-exact with the host and the F411 on a core that has no unaligned access.

**On the MLP, int8 against software float is 22×** (33.0k vs 712k cycles at `-O2`). That is not int8 against int8, since emlearn has no int8 MLP; it is what each library offers for this network on a part without an FPU.

**The random forest went from a 3× loss to a tie.** The first run walked the forest with `QDecisionTree`: about 20 Thumb-1 instructions per node and a full 12-byte node per leaf, against emlearn's generated load, compare-immediate and branch. [`QStaticDecisionTree`]({{ site.baseurl }}/architectures/decision-trees#compile-time-models) (#223) generates that same code from the same node pool at compile time, so the pool is not stored at all. It is 2.9× faster than the walk in about half its size. With `uint8_t` features, which load with Thumb-1's immediate-offset `LDRB` instead of a register-offset `LDRSB`, it beats emlearn at `-O2` (872 vs 917 cycles), is within 0.5% at `-O3` (845 vs 841), is smaller at `-O3` (7.9 vs 8.3 KB) and uses less RAM (112 vs 184 B). When flash matters more than cycles, the compact policy packs the same pool at compile time into 4-byte decision nodes in pre-order and a deduplicated int8 leaf table: the whole forest is 3.0 KB, half of emlearn's smallest and a quarter of the walk, at 1.5× the walk's speed and about half of inline's. Against emlearn's data-driven `loadable` method it is 3.7–3.9× faster in 42% of the flash. The model is also checked when it compiles: a split on a feature the input does not have, or a child index outside the pool, is a build error instead of an out-of-bounds read.

**The MLP is 7–9% faster than in the previous measurement, after first being 36% slower.** Re-measured at `6a8b522`, it took 48.6k cycles at `-O2` against 35.6k before. Building each library change since then on the board placed the jump at [#230](https://github.com/danmcleran/tinymind/pull/230), which grew `Requantizer::apply`'s saturating path (scale ratios of 0.25 and up, which the MLP never takes). The dense layers call `apply` once per output, so that path's code sat in the output loop, and on Thumb-1, whose instructions mostly reach eight registers, GCC at `-O2` spilled the dot product beside it. [#275](https://github.com/danmcleran/tinymind/pull/275) checks once per layer whether every requantizer takes the one-rounding path and then runs a copy of the loop without the saturating code, for GCC on Thumb-1 only: 33.0k cycles at `-O2` and 32.6k at `-O3`, against 35.6k and 36.0k before #230, bit-exact, for 0.9 KB more flash at `-O2`.

**M0+ against M4F, same code.** At `-O3` the MLP takes 1.6× the cycles of the F411's scalar build and the CNN 2.9×. Requantization's 64-bit multiply becomes a library call (`__aeabi_lmul`) on ARMv6-M, and the 3×3 stem is 69% of the CNN, so those are where M0+ tuning would start.

## tinymind under Clang

Every table above is GCC. `make TOOLCHAIN=clang` rebuilds the tinymind images with Clang 18.1.3 instead (TFLM and emlearn stay on GCC), linked against the same newlib and startup code, and every one of them is bit-exact with the GCC build on both boards.

Two settings matter. **`-munaligned-access`**: Clang's bare-metal Arm target assumes strict alignment unless told otherwise, so without it every 32-bit load in the SMLAD backend becomes four byte loads, and the MLP with SMLAD took 23,098 cycles, slower than Clang's own scalar build (22,054). The bench's `clang-arm-cxx` passes it on every core that has unaligned access; the Cortex-M0+ does not. **`-fshort-enums`**: GCC's `arm-none-eabi` target uses short enums and newlib is built that way, so a Clang object without it does not match the C library's ABI.

`-O3`, cycles:

| Board | Model | GCC 13.2.1 | Clang 18.1.3 | Clang vs GCC |
|---|---|---:|---:|---:|
| F411 | MLP, scalar | 21,002 | 22,024 | +4.9% |
| | MLP, +SMLAD | 10,227 | 12,280 | +20.1% |
| | KWS CNN, scalar | 254,766 | 292,351 | +14.8% |
| | KWS CNN, +SMLAD | 181,853 | 222,330 | +22.3% |
| | MLPerf keyword spotting, +SMLAD | 8,011,750 | 10,300,182 | +28.6% |
| | MLPerf image classification, +SMLAD | 26,368,775 | 38,141,841 | +44.6% |
| L031 | MLP | 32,644 | 39,339 | +20.5% |
| | KWS CNN | 740,279 | 808,981 | +9.3% |
| | Random forest, compile-time inline (uint8) | 845 | 930 | +10.1% |
| | Random forest, run-time walk | 2,524 | 2,310 | −8.5% |

**Build tinymind with GCC where you can.** Clang is slower on every network, by 3–45%, and most on the SMLAD builds. Built with Clang, tinymind no longer beats TFLM with CMSIS-NN (built with GCC) on the MLPerf models: 10.3M against 9.6M cycles on keyword spotting and 38.1M against 37.4M on image classification. It still beats it on the two models built for this benchmark (12.3k against 14.0k on the MLP, 222k against 576k on the CNN). The main gaps are not yet explained; one visible difference is that Clang calls `Requantizer::apply` out of line where GCC inlines it. The Cortex-M0+ fix in #275 is not applied under Clang, which ran the MLP 7–18% slower with it.

## What changed since the first run

The first runs (tinymind before #192) exposed four weak spots. All four are fixed, along with five follow-ups. Every int8 output hash is unchanged, so every fix is bit-exact.

| tinymind | First run | Now |
|---|---:|---:|
| MLP +SMLAD `-O2` | 24,591 | 10,167 |
| MLP +SMLAD `-O3` | 12,278 | 10,227 |
| MLP scalar `-O3` | 33,154 | 21,002 |
| CNN +SMLAD `-O2` | 831,570 | 195,841 |
| CNN +SMLAD `-O3` | 422,449 | 181,853 |
| CNN scalar `-O3` | 445,579 | 254,766 |

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
- **tinymind:** header-only, with nothing to build or link. It has no `.tflite` importer. The bridges that copied the TFLite int8 parameters into tinymind layers (`gen/make_models.py`, and `gen/make_mlperf.py` for the MLPerf Tiny models) were written for this benchmark. tinymind ships [PyTorch, ONNX and tree-ensemble importers](https://github.com/danmcleran/tinymind/tree/master/apps); the tree importer covers scikit-learn, XGBoost and LightGBM.

## Reproduce it

Everything is in [`benchmarks/framework_comparison_f411`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411): bare-metal board support, a timing harness, one backend per framework, the model generator, and the raw results behind these tables. With `arm-none-eabi-gcc`, `openocd` and a NUCLEO-F411RE attached:

```
cd benchmarks/framework_comparison_f411
make deps            # Python venv + TFLM clone at the pinned revision + TFLM libraries
make host-check      # tinymind int8 outputs reproduce the expected hashes
make LEVEL=O3 run    # build, flash and capture every image
```

For the MLPerf Tiny reference models, `make mlperf-models` downloads them and `make SUITE=mlperf LEVEL=O3 run` writes `results/raw_mlperf_O3.txt`.

On a NUCLEO-L031K6, `make BOARD=l031 LEVEL=O3 run` builds and runs the Cortex-M0+ images (`results/raw_l031_O3.txt`), and `make BOARD=l031 fit` reports what the images that do not fit would need.

Adding `TOOLCHAIN=clang` to any of these runs builds the tinymind images with Clang and writes `results/raw_*_clang.txt`.

The generated model headers are committed, so the firmware builds without TensorFlow; `make deps` is only needed for the TFLM and emlearn images.

## A note on host references

TensorFlow's pip `tf.lite.Interpreter` differs from all three on-device int8 paths by ±1 LSB on 35 of 3,600 MLP outputs: its fully-connected kernel rounds once where TFLM, CMSIS-NN and tinymind round twice. If you validate an MCU int8 pipeline against the desktop interpreter, expect occasional last-bit differences that are not bugs.
