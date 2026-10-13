# Framework comparison on NUCLEO-F411RE: TinyMind vs TFLM vs emlearn

On-target benchmark of TinyMind against TensorFlow Lite for Microcontrollers (TFLM, now LiteRT for
Microcontrollers) with CMSIS-NN and with reference kernels, and against emlearn. The board is a
NUCLEO-F411RE: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on.
The toolchain is arm-none-eabi-gcc 13.2.1 with the hard-float ABI. Cycles come from the DWT cycle
counter, as the median of 33 warm inferences (spread < 0.1%).

Write-up: https://danmcleran.github.io/tinymind/benchmarks.html

Versions: TinyMind (this repo; results below at `3109e3b`) · TFLM @ `18811372` (2026-10-12) with its pinned CMSIS-NN ·
emlearn 0.23.2 · TensorFlow 2.21 (converter only) · scikit-learn 1.9.1.

This folder lives outside `examples/` on purpose. The top-level build and `check-clang-examples` build
every Makefile under `examples/` with the host compiler, and this one targets Arm only. CI does not build
it.

## Method

* One float model per network is the source of truth (`gen/make_models.py`). The MLP is trained on
  scikit-learn digits. The KWS CNN has random weights, so it measures speed and exactness only.
* TFLite post-training int8 quantization with TF 2.21 defaults: per-channel weights for both conv and dense.
* The **same int8 weights, biases and requantizer multipliers** are lifted out of the `.tflite` file and
  emitted as TinyMind `QConv2DPerChannel` / `QDepthwiseConv2D` / `QPointwiseConv2DPerChannel` / pool layers
  (`gen/out/*_tinymind.hpp`). These are `const` aggregates in flash, so there is no setup step. Both
  runtimes run identical integer math, so their int8 outputs must hash-match on the board. They do, for
  every sample in every build. `make host-check` reproduces the expected hashes on the host.
* emlearn gets the same scikit-learn float MLP. Its final softmax is patched to identity at build time so
  that every backend returns logits; the argmax is unchanged.
* `LEVEL=O2|O3` sets the optimization level for every framework together. TinyMind and emlearn translation
  units get `-O2`/`-O3`. TFLM links its own default release build (kernels `-O2`, core `-Os`) or a rebuild
  with every level forced to `-O3`.
* `flash` = image − empty-harness image − embedded test-set bytes, i.e. runtime code + weights.
  `RAM` = (.data + .bss − baseline) + (peak stack − baseline). TFLM tensor arenas are the minimum
  `AllocateTensors()` accepts, bisected on the board (`make probe`).
* `gen/out/` is committed: the generator is deterministic (re-running it reproduces every file byte for
  byte), and committing the output means the firmware builds without TensorFlow.

## Results

The tables below come from `results/raw_O2.txt` and `results/raw_O3.txt`, measured at TinyMind `3109e3b`
(after #192 and #195; see finding 3 for what changed since the first run).

## MLP 64-32-16-10, sklearn digits (360 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 31,301 | 33,154 | 332 | 4,400 / 4,828 | 148 | 96.11% | yes |
| TinyMind int8 (+SMLAD) | 11,093 | 11,160 | 112 | 5,196 / 5,364 | 124 | 96.11% | yes |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 | 36,736 / 40,624 | 2,600 | 96.11% | yes |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 | 25,664 / 29,552 | 2,152 | 96.11% | yes |
| TFLM float32 | 27,902 | 28,059 | 281 | 43,228 / 47,276 | 1,824 | 96.11% | n/a (float) |
| emlearn float32 | 40,527 | 40,529 | 405 | 14,392 / 14,736 | 892 | 96.11% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| fc0 | 22,881 | 22,701 | 7,696 | 7,724 | 8,592 | 8,556 |
| fc1 | 6,110 | 6,136 | 2,473 | 2,473 | 3,421 | 3,375 |
| fc2 | 2,089 | 1,434 | 1,124 | 1,127 | 2,076 | 2,035 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 619,805 | 428,938 | 4,289 | 2,612 / 3,840 | 4,884 | yes |
| TinyMind int8 (+SMLAD) | 409,686 | 384,029 | 3,840 | 2,856 / 3,644 | 4,880 | yes |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5,764 | 84,488 / 91,640 | 6,288 | yes |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24,356 | 41,088 / 46,192 | 5,668 | yes |
| TFLM float32 | 2,136,067 | 2,159,717 | 21,597 | 84,324 / 91,604 | 15,128 | n/a (float) |

### Per-layer cycles (kws, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv3x3 | 376,561 | 280,261 | 235,866 | 259,863 | 434,553 | 435,412 |
| maxpool | 14,474 | 10,161 | 14,477 | 10,158 | 35,807 | 36,697 |
| dwconv3x3 | 71,238 | 57,941 | 71,243 | 58,219 | 39,208 | 40,918 |
| pwconv | 113,619 | 82,627 | 63,036 | 67,487 | 53,212 | 53,866 |
| avgpool | 7,222 | 2,281 | 7,221 | 2,282 | 8,875 | 7,917 |
| fc | 2,075 | 1,433 | 1,127 | 1,132 | 2,097 | 2,041 |


TFLM setup (`AllocateTensors`) costs 133k–163k cycles (1.3–1.6 ms) once at boot. TinyMind has no setup step.

## Findings

1. **Bit-exact.** TinyMind and TFLM (both CMSIS-NN and reference kernels) produce identical int8 outputs on
   the board, so accuracy is identical (96.11%). The pip TFLite *host* interpreter differs by ±1 LSB on
   35/3600 MLP outputs: its fully-connected kernel uses single rounding, while TFLM, CMSIS-NN and TinyMind
   use double rounding.
2. **TinyMind+SMLAD is faster than TFLM+CMSIS-NN at both levels:** MLP 11.2k vs 14.0k cycles at `-O3`
   (−20%) and 11.1k vs 14.1k at `-O2`; KWS 384k vs 576k at `-O3` (−33%) and 410k vs 573k at `-O2` (−29%).
   The scalar path still gains from `-O3` (KWS 620k vs 429k), so `-O3` remains the recommended flag.
3. **What changed since the first run** (TinyMind+SMLAD, bit-exact, hashes unchanged):
   MLP `-O2` 24,591 → 11,093, MLP `-O3` 12,278 → 11,160, KWS `-O2` 831,570 → 409,686, KWS `-O3`
   422,449 → 384,029.
   - #192: at `-O2` the SMLAD inner loop stayed rolled inside each layer's output loop, where the
     requantizer's hoisted constants left it about three scratch registers. It ran 2 SMLADs per iteration,
     round-tripped the loaded words through the stack and reloaded its loop bound every pass, and with
     identical kernels the MLP measured anywhere from 14.5k to 24.6k cycles. The loop is now
     pragma-unrolled (so `-O2` compiles it like `-O3`) and the zero point is folded into `SXTAB16`.
   - #195: the conv layers called the dot product once per kernel tap, so the Cin=1 3×3 stem ran nine
     length-1 dot products per output. Each pixel's kernel window is now gathered once (≤ 64 B; larger
     windows run one dot product per kernel row) and reused by every filter.
4. **Per layer (KWS):** TinyMind wins the Cin=1 3×3 conv (236k at `-O2`, 260k at `-O3`, vs 435k;
   CMSIS-NN's im2col path is poor for single-channel input), max/avg pooling (3–4× faster) and the dense
   head. It still loses depthwise conv (58k vs 41k at `-O3`, a scalar loop) and pointwise conv (67k vs 54k).
   Those two kernels are the next optimization targets.
5. **Code + weights: 10–30× smaller.** TinyMind is 4–5.4 KB (MLP) and 3–4 KB (KWS) including weights. TFLM is
   26–41 KB (MLP) and 41–92 KB (KWS): the interpreter, the flatbuffer parser and every CMSIS-NN type
   variant that the kernel dispatch references (s4/s16 conv, ...) all get linked. emlearn is 14 KB, almost
   all of it float weights.
6. **RAM:** MLP: TinyMind ~130 B vs TFLM 2.2–2.6 KB. KWS: TinyMind 4.9 KB vs TFLM 5.7–6.3 KB. TinyMind keeps
   every intermediate buffer static (no reuse); TFLM's arena planner reuses buffers but carries
   interpreter state. A ping-pong buffer pair would bring the TinyMind KWS model to about 3.6 KB.
7. **Float:** TFLM float32 (27.9k) beats emlearn (40.5k) on the MLP; emlearn walks its weights
   column-strided, but its code is tiny. TFLM float KWS is 2.1M cycles because this CMSIS-NN build has its
   float kernels disabled and falls back to the reference float kernels.

## Integration friction observed

* **emlearn:** `pip install emlearn`, `emlearn.convert(model).save()`, include one header. Minutes.
  MLPs and trees only, float only.
* **TFLM:** `git clone` + `make ... microlite`, which downloads CMSIS, CMSIS-NN and flatbuffers. Converting
  needs full TensorFlow (~600 MB). Pitfalls hit here:
  - there is no pip `tflite-micro`
  - the schema header lives in `tensorflow/lite/micro/schema/`
  - the lib is built with `-funsigned-char`
  - a Keras `Flatten` with a dynamic batch adds SHAPE/STRIDED_SLICE/PACK ops unless `batch_size=1`
  - `arena_used_bytes()` under-reports the minimum arena by 100–300 B
  - the per-op profiler is compiled out of release builds (`TF_LITE_STRIP_ERROR_STRINGS`), so it needs a
    `release_with_logs` lib
* **TinyMind:** header-only and has no `.tflite` importer. The TFLite→TinyMind bridge in
  `gen/make_models.py` was written for this benchmark (TinyMind ships PyTorch/ONNX importers in `apps/`).

## Reproduce

Needs `arm-none-eabi-gcc`, `openocd`, a NUCLEO-F411RE on `/dev/ttyACM0`, and Python 3 with `venv`.

```
cd benchmarks/framework_comparison_f411
make deps                      # .venv (tensorflow-cpu, emlearn, scikit-learn, numpy),
                               # TFLM clone at the pinned revision, six TFLM libs
make host-check                # TinyMind int8 outputs match the expected hashes
make LEVEL=O2 run              # build + flash + capture -> results/raw_O2.txt
make LEVEL=O3 run              # -> results/raw_O3.txt
.venv/bin/python -I report.py  # -> results/REPORT.md
```

`make models` regenerates `gen/out/` from scratch (needs the venv). `make probe` re-bisects the TFLM arena
sizes; update the `ARENA_*` values in the Makefile if they change.

## Layout

| Path | What |
|---|---|
| `board/` | Bare-metal F411 support: clock, USART2 on the ST-LINK VCP, DWT, stack painting, linker script |
| `fw/harness.cpp` | Times one backend and prints a `key=value` report |
| `fw/be_*.c*` | One backend per framework: TinyMind, TFLM, emlearn, empty baseline |
| `gen/make_models.py` | Trains/quantizes the models and emits every framework's headers into `gen/out/` |
| `host/tm_check.cpp` | Host-side TinyMind hash check |
| `report.py` | `results/raw_*.txt` → `results/REPORT.md` |
