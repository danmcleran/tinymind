# Framework comparison on NUCLEO-F411RE: TinyMind vs TFLM vs emlearn

On-target benchmark of TinyMind against TensorFlow Lite for Microcontrollers (TFLM, now LiteRT for
Microcontrollers) with CMSIS-NN and with reference kernels, and against emlearn. The board is a
NUCLEO-F411RE: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on.
The toolchain is arm-none-eabi-gcc 13.2.1 with the hard-float ABI. Cycles come from the DWT cycle
counter, as the median of 33 warm inferences (spread < 0.1%).

Write-up: https://danmcleran.github.io/tinymind/benchmarks.html

Versions: TinyMind (this repo; results below at `b02e92d`) · TFLM @ `18811372` (2026-10-12) with its pinned CMSIS-NN ·
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

The tables below come from `results/raw_O2.txt` and `results/raw_O3.txt`, measured at TinyMind `b02e92d`
(after #192, #195, #198, #199, #202, #204, #208, #210 and #211; see finding 3 for what changed since the first run).

## MLP 64-32-16-10, sklearn digits (360 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 20,639 | 20,486 | 205 | 5,296 / 5,464 | 152 | 96.11% | yes |
| TinyMind int8 (+SMLAD) | 10,305 | 10,241 | 102 | 5,380 / 5,548 | 124 | 96.11% | yes |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 | 36,736 / 40,624 | 2,600 | 96.11% | yes |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 | 25,664 / 29,552 | 2,152 | 96.11% | yes |
| TFLM float32 | 27,902 | 28,059 | 281 | 43,228 / 47,276 | 1,824 | 96.11% | n/a (float) |
| emlearn float32 | 40,527 | 40,529 | 405 | 14,392 / 14,736 | 892 | 96.11% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| fc0 | 15,096 | 15,097 | 7,344 | 7,345 | 8,592 | 8,556 |
| fc1 | 4,147 | 4,151 | 2,221 | 2,225 | 3,421 | 3,375 |
| fc2 | 1,279 | 1,280 | 948 | 949 | 2,076 | 2,035 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| TinyMind int8 (scalar) | 266,451 | 254,900 | 2,549 | 4,996 / 6,848 | 5,332 | yes |
| TinyMind int8 (+SMLAD) | 192,965 | 181,568 | 1,816 | 4,692 / 6,364 | 5,296 | yes |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5,764 | 84,488 / 91,640 | 6,288 | yes |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24,356 | 41,088 / 46,192 | 5,668 | yes |
| TFLM float32 | 2,136,067 | 2,159,717 | 21,597 | 84,324 / 91,604 | 15,128 | n/a (float) |

### Per-layer cycles (kws, each layer timed separately)

| Layer | TinyMind scalar O2 | TinyMind scalar O3 | TinyMind +SMLAD O2 | TinyMind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv3x3 | 181,968 | 173,164 | 132,155 | 127,756 | 434,553 | 435,412 |
| maxpool | 10,166 | 10,156 | 10,162 | 10,160 | 35,807 | 36,697 |
| dwconv3x3 | 21,413 | 21,054 | 21,415 | 21,047 | 39,208 | 40,918 |
| pwconv | 48,096 | 44,258 | 22,941 | 25,865 | 53,212 | 53,866 |
| avgpool | 2,266 | 2,282 | 2,267 | 2,281 | 8,875 | 7,917 |
| fc | 1,284 | 1,281 | 948 | 946 | 2,097 | 2,041 |


TFLM setup (`AllocateTensors`) costs 133k–163k cycles (1.3–1.6 ms) once at boot. TinyMind has no setup step.

## Findings

1. **Bit-exact.** TinyMind and TFLM (both CMSIS-NN and reference kernels) produce identical int8 outputs on
   the board, so accuracy is identical (96.11%). The pip TFLite *host* interpreter differs by ±1 LSB on
   35/3600 MLP outputs: its fully-connected kernel uses single rounding, while TFLM, CMSIS-NN and TinyMind
   use double rounding.
2. **TinyMind+SMLAD is faster than TFLM+CMSIS-NN on every layer, at both levels:** MLP 10.2k vs 14.0k cycles
   at `-O3` (−27%); KWS 182k vs 576k at `-O3` (−68%) and 193k vs 573k at `-O2` (−66%). The portable scalar
   build still runs KWS 56% faster than CMSIS-NN (255k); on the MLP it (20.5k) sits between CMSIS-NN and
   TFLM's reference kernels (31.4k). The MLP is within 1% across `-O2`/`-O3`; the KWS model is 6% slower at
   `-O2` (+SMLAD) and 5% slower scalar. Use `-O3`.
3. **What changed since the first run** (all bit-exact, hashes unchanged): MLP+SMLAD `-O2` 24,591 → 10,305,
   MLP scalar `-O3` 33,154 → 20,486, KWS+SMLAD `-O2` 831,570 → 192,965, KWS+SMLAD `-O3` 422,449 → 181,568.
   - #192: the SMLAD inner loop stayed rolled at `-O2` inside each layer's output loop and spilled to the
     stack (2 SMLADs per iteration; the MLP measured 14.5k–24.6k cycles across builds with identical
     kernels). Now pragma-unrolled, with the zero point folded into `SXTAB16`.
   - #195: conv called the dot product once per kernel tap (nine length-1 dots per output for the Cin=1
     stem). Each pixel's kernel window is now gathered once (≤ 64 B; larger windows run one dot per
     kernel row) and reused by every filter.
   - #198: pointwise now runs filter-outer (weights, bias and requantizer stay in locals across pixels;
     single-pixel/dense layers keep the per-output form), and depthwise runs channel-outer with the zero
     point folded into the bias. Depthwise 58k → 31k, pointwise 67k → 44k at `-O3`.
   - #199: the portable scalar dot product got the same unroll hint (scalar MLP 31k–33k → 21.5k).
   - #202: the conv patch path gathers a whole tile of windows (up to `TINYMIND_CONV_TILE_MAX_BYTES`,
     default 256 B; a whole output row for this stem) and sweeps it filter-outer, so each filter's
     weights, bias and requantizer are loaded once per tile instead of once per pixel. Stem 260k → 177k
     at `-O3`, for about 0.2 KB more peak stack.
   - #204: the pooling window loops got the same unroll hint. At `-O2`, max-pool 14.5k → 10.2k and the
     global average pool 7.2k → 2.3k, now level with `-O3`.
   - #208: `Requantizer::apply` fuses gemmlowp's two roundings (rounding high multiply, then rounding
     shift) into one exact 64-bit rounding when multiplier ≥ 0 and shift ≥ 2 (every layer here): an
     `SMLAL` and an `ASR` on the M4 instead of about 30 instructions with seven stack reloads. Stem 177k →
     158k, depthwise 31k → 23k, pointwise 44k → 39k at `-O3`; MLP 11.2k → 10.2k.
   - #210: the stem's output loop reloaded the filter's weights after every int8 output store (the
     store may alias the weights) and spilled six requantizer values. Weights now go in a local, and
     `detail::FastRequantizer` decodes the requantizer once per filter (zero point folded into the
     rounding constant, clamp as one `SSAT`; fused-activation clamps keep the general path). Stem
     158k → 128k at `-O3`, 190k → 132k at `-O2`.
   - #211: the same two fixes for pointwise and depthwise. Pointwise 39k → 26k at `-O3` (41k → 23k at
     `-O2`), depthwise 23k → 21k.
4. **Per layer (KWS):** TinyMind wins every layer. The Cin=1 3×3 stem is still about two thirds of the
   model's time (128k at `-O3`, 29% of CMSIS-NN's 435k).
5. **Code + weights: 7–18× smaller than TFLM with CMSIS-NN.** TinyMind is 5.3–5.5 KB (MLP) and 4.7–6.8 KB (KWS) including weights. TFLM is
   26–41 KB (MLP) and 41–92 KB (KWS): the interpreter, the flatbuffer parser and every CMSIS-NN type
   variant that the kernel dispatch references (s4/s16 conv, ...) all get linked. emlearn is 14 KB, almost
   all of it float weights.
6. **RAM:** MLP: TinyMind ~130 B vs TFLM 2.2–2.6 KB. KWS: TinyMind 5.3 KB vs TFLM 5.7–6.3 KB. TinyMind keeps
   every intermediate buffer static (no reuse); TFLM's arena planner reuses buffers but carries
   interpreter state. A ping-pong buffer pair (2,592 B + 784 B instead of 4,432 B) would save about 1 KB.
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
