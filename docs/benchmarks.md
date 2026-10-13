---
title: On-Target Benchmarks
layout: default
nav_order: 14
---

# On-Target Benchmarks: TinyMind vs TFLM vs emlearn

TinyMind, TensorFlow Lite for Microcontrollers (TFLM, now LiteRT for Microcontrollers) and emlearn were run on the same microcontroller. Every framework executed the same two models, and the int8 frameworks used the same int8 weights. Everything was measured on the board; nothing was simulated.

{: .note }
Measured 2026-10-13 with the code in [`benchmarks/framework_comparison_f411`](https://github.com/danmcleran/tinymind/tree/master/benchmarks/framework_comparison_f411). These are numbers from one board, one compiler and two models. They describe this setup, not every workload.

## Setup

| | |
|---|---|
| Board | NUCLEO-F411RE: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on |
| Compiler | arm-none-eabi-gcc 13.2.1, `-mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16` |
| Timing | DWT cycle counter; median of 33 warm inferences (spread under 0.1%) |
| TinyMind | master @ `bb332c2`, int8 layer family, with and without the `TINYMIND_ENABLE_SIMD_ARM_DSP` (SMLAD) backend |
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
| TinyMind int8 (scalar) | 31,301 | 33,154 | 332 µs | 4.4–4.8 KB | 148 B | 96.11% |
| **TinyMind int8 (+SMLAD)** | 24,591 | **12,278** | **123 µs** | 4.5–5.6 KB | 124 B | 96.11% |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 µs | 36.7–40.6 KB | 2,600 B | 96.11% |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 µs | 25.7–29.6 KB | 2,152 B | 96.11% |
| TFLM float32 | 27,902 | 28,059 | 281 µs | 43.2–47.3 KB | 1,824 B | 96.11% |
| emlearn float32 | 40,527 | 40,529 | 405 µs | 14.4–14.7 KB | 892 B | 96.11% |

### KWS-style CNN

| Framework | Cycles `-O2` | Cycles `-O3` | Time `-O3` | Code + weights | RAM |
|---|---:|---:|---:|---:|---:|
| TinyMind int8 (scalar) | 795,368 | 445,579 | 4.46 ms | 2.6–3.8 KB | 4,876 B |
| **TinyMind int8 (+SMLAD)** | 831,570 | **422,449** | **4.22 ms** | 2.7–3.7 KB | 4,864 B |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5.76 ms | 84.5–91.6 KB | 6,288 B |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24.4 ms | 41.1–46.2 KB | 5,668 B |
| TFLM float32 | 2,136,067 | 2,159,717 | 21.6 ms | 84.3–91.6 KB | 15,128 B |

Code + weights ranges are `-O2` / `-O3`.

TFLM also spends 133k–163k cycles (1.3–1.6 ms) in `AllocateTensors()` once at boot. TinyMind has no setup step, because its layers are `const` aggregates initialized at compile time.

### Per-layer cycles, KWS CNN

Each layer was timed separately, behind its own non-inlined call.

| Layer | TinyMind +SMLAD `-O2` | TinyMind +SMLAD `-O3` | TFLM CMSIS-NN `-O2` | TFLM CMSIS-NN `-O3` |
|---|---:|---:|---:|---:|
| conv 3×3, 1→8 | 467,486 | **279,873** | 434,553 | 435,412 |
| max-pool 2×2 | 14,477 | **10,158** | 35,807 | 36,697 |
| depthwise 3×3 | 71,239 | 58,219 | 39,208 | **40,918** |
| pointwise 8→16 | 76,690 | 78,425 | 53,212 | **53,866** |
| average-pool | 7,221 | **2,282** | 8,875 | 7,917 |
| dense 16→10 | 1,264 | **1,186** | 2,097 | 2,041 |

## What the numbers say

**At `-O3`, TinyMind is faster and an order of magnitude smaller.** With the SMLAD backend, the MLP runs 12% faster than TFLM with CMSIS-NN and the CNN runs 27% faster. Code + weights are 3–6 KB against 37–92 KB, because there is no interpreter or flatbuffer parser, and no unused kernel variants get linked in.

**At `-O2`, TinyMind loses the MLP, and its speed becomes unpredictable.** TinyMind's kernels are templates compiled in your own translation unit, so the optimizer's choices are part of its performance. In the disassembly, the `-O2` inner loop of the first MLP layer issues 2 SMLADs per iteration, stores each loaded word to the stack and reads it back, and reloads its loop bound from memory on every pass. At `-O3` the same loop is fully unrolled: 32 SMLADs and no stack traffic. How badly `-O2` degrades depends on register allocation, which shifts with the surrounding code. Across builds with identical kernels, the `-O2` MLP measured anywhere from 14.5k to 24.6k cycles, while `-O3` stayed within 1%. CMSIS-NN's hand-written kernels ship precompiled, so they barely move between levels. **Build TinyMind at `-O3`**, which is also what its own release Makefiles use.

**Where TinyMind still trails.** Depthwise convolution (58k vs 41k cycles) and pointwise convolution (78k vs 54k) are slower than CMSIS-NN. The depthwise kernel is a scalar loop, and SMLAD brings no gain on the pointwise layer at `-O3`. These two kernels are the next optimization targets. TinyMind wins the single-input-channel 3×3 convolution (280k vs 435k cycles), where CMSIS-NN's im2col path does poorly, and it is 3–4× faster on pooling.

**RAM is close on the CNN.** TinyMind keeps every intermediate activation buffer static. TFLM's arena planner reuses buffers but carries interpreter state. Alternating between two buffers would bring TinyMind's CNN to about 3.6 KB.

**Among the float runtimes,** TFLM float32 is faster than emlearn on the MLP. emlearn has the smallest float footprint, and almost all of it is the weights themselves.

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
