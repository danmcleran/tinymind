# Framework comparison on NUCLEO-F411RE: tinymind vs TFLM vs emlearn

On-target benchmark of tinymind against TensorFlow Lite for Microcontrollers (TFLM, now LiteRT for
Microcontrollers) with CMSIS-NN and with reference kernels, and against emlearn. The board is a
NUCLEO-F411RE: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on.
The toolchain is arm-none-eabi-gcc 13.2.1 with the hard-float ABI. Cycles come from the DWT cycle
counter, as the median of 33 warm inferences (spread < 0.1%).

Write-up: https://danmcleran.github.io/tinymind/benchmarks.html

Versions: tinymind (this repo; results below at `b02e92d`) · TFLM @ `18811372` (2026-10-12) with its pinned CMSIS-NN ·
emlearn 0.23.2 · TensorFlow 2.21 (converter only) · scikit-learn 1.9.1.

A second board, a Cortex-M0+ NUCLEO-L031K6 with 32 KB of flash, runs the same comparison; see
[Cortex-M0+: NUCLEO-L031K6](#cortex-m0-nucleo-l031k6).

This folder lives outside `examples/` on purpose. The top-level build and `check-clang-examples` build
every Makefile under `examples/` with the host compiler, and this one targets Arm only. CI does not build
it.

## Method

* One float model per network is the source of truth (`gen/make_models.py`). The MLP is trained on
  scikit-learn digits. The KWS CNN has random weights, so it measures speed and exactness only.
* TFLite post-training int8 quantization with TF 2.21 defaults: per-channel weights for both conv and dense.
* The **same int8 weights, biases and requantizer multipliers** are lifted out of the `.tflite` file and
  emitted as tinymind `QConv2DPerChannel` / `QDepthwiseConv2D` / `QPointwiseConv2DPerChannel` / pool layers
  (`gen/out/*_tinymind.hpp`). These are `const` aggregates in flash, so there is no setup step. Both
  runtimes run identical integer math, so their int8 outputs must hash-match on the board. They do, for
  every sample in every build. `make host-check` reproduces the expected hashes on the host.
* emlearn gets the same scikit-learn float MLP (it has no int8 MLP). Its final softmax is patched to identity at build time so
  that every backend returns logits; the argmax is unchanged.
* `LEVEL=O2|O3` sets the optimization level for every framework together. tinymind and emlearn translation
  units get `-O2`/`-O3`. TFLM links its own default release build (kernels `-O2`, core `-Os`) or a rebuild
  with every level forced to `-O3`.
* `flash` = image − empty-harness image − embedded test-set bytes, i.e. runtime code + weights.
  `RAM` = (.data + .bss − baseline) + (peak stack − baseline). TFLM tensor arenas are the minimum
  `AllocateTensors()` accepts, bisected on the board (`make probe`).
* `gen/out/` is committed: the generator is deterministic (re-running it reproduces every file byte for
  byte), and committing the output means the firmware builds without TensorFlow.

## Results

The tables below come from `results/raw_O2.txt` and `results/raw_O3.txt`, measured at tinymind `b02e92d`
(after #192, #195, #198, #199, #202, #204, #208, #210 and #211; see finding 3 for what changed since the first run).

## MLP 64-32-16-10, sklearn digits (360 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 20,639 | 20,486 | 205 | 5,296 / 5,464 | 152 | 96.11% | yes |
| tinymind int8 (+SMLAD) | 10,305 | 10,241 | 102 | 5,380 / 5,548 | 124 | 96.11% | yes |
| TFLM int8 (CMSIS-NN) | 14,075 | 14,030 | 140 | 36,736 / 40,624 | 2,600 | 96.11% | yes |
| TFLM int8 (reference) | 32,084 | 31,411 | 314 | 25,664 / 29,552 | 2,152 | 96.11% | yes |
| TFLM float32 | 27,902 | 28,059 | 281 | 43,228 / 47,276 | 1,824 | 96.11% | n/a (float) |
| emlearn float32 | 40,527 | 40,529 | 405 | 14,392 / 14,736 | 892 | 96.11% | n/a (float) |

### Per-layer cycles (mlp, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| fc0 | 15,096 | 15,097 | 7,344 | 7,345 | 8,592 | 8,556 |
| fc1 | 4,147 | 4,151 | 2,221 | 2,225 | 3,421 | 3,375 |
| fc2 | 1,279 | 1,280 | 948 | 949 | 2,076 | 2,035 |

## KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @100MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 266,451 | 254,900 | 2,549 | 4,996 / 6,848 | 5,332 | yes |
| tinymind int8 (+SMLAD) | 192,965 | 181,568 | 1,816 | 4,692 / 6,364 | 5,296 | yes |
| TFLM int8 (CMSIS-NN) | 573,396 | 576,368 | 5,764 | 84,488 / 91,640 | 6,288 | yes |
| TFLM int8 (reference) | 2,459,339 | 2,435,580 | 24,356 | 41,088 / 46,192 | 5,668 | yes |
| TFLM float32 | 2,136,067 | 2,159,717 | 21,597 | 84,324 / 91,604 | 15,128 | n/a (float) |

### Per-layer cycles (kws, each layer timed separately)

| Layer | tinymind scalar O2 | tinymind scalar O3 | tinymind +SMLAD O2 | tinymind +SMLAD O3 | TFLM CMSIS-NN O2 | TFLM CMSIS-NN O3 |
|---|---:|---:|---:|---:|---:|---:|
| conv3x3 | 181,968 | 173,164 | 132,155 | 127,756 | 434,553 | 435,412 |
| maxpool | 10,166 | 10,156 | 10,162 | 10,160 | 35,807 | 36,697 |
| dwconv3x3 | 21,413 | 21,054 | 21,415 | 21,047 | 39,208 | 40,918 |
| pwconv | 48,096 | 44,258 | 22,941 | 25,865 | 53,212 | 53,866 |
| avgpool | 2,266 | 2,282 | 2,267 | 2,281 | 8,875 | 7,917 |
| fc | 1,284 | 1,281 | 948 | 946 | 2,097 | 2,041 |


TFLM setup (`AllocateTensors`) costs 133k–163k cycles (1.3–1.6 ms) once at boot. tinymind has no setup step.

## Findings

1. **Bit-exact.** tinymind and TFLM (both CMSIS-NN and reference kernels) produce identical int8 outputs on
   the board, so accuracy is identical (96.11%). The pip TFLite *host* interpreter differs by ±1 LSB on
   35/3600 MLP outputs: its fully-connected kernel uses single rounding, while TFLM, CMSIS-NN and tinymind
   use double rounding.
2. **tinymind+SMLAD is faster than TFLM+CMSIS-NN on every layer, at both levels:** MLP 10.2k vs 14.0k cycles
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
4. **Per layer (KWS):** tinymind wins every layer. The Cin=1 3×3 stem is still about two thirds of the
   model's time (128k at `-O3`, 29% of CMSIS-NN's 435k).
5. **Code + weights: 7–18× smaller than TFLM with CMSIS-NN.** tinymind is 5.3–5.5 KB (MLP) and 4.7–6.8 KB (KWS) including weights. TFLM is
   26–41 KB (MLP) and 41–92 KB (KWS): the interpreter, the flatbuffer parser and every CMSIS-NN type
   variant that the kernel dispatch references (s4/s16 conv, ...) all get linked. emlearn is 14 KB, almost
   all of it float weights.
6. **RAM:** MLP: tinymind ~130 B vs TFLM 2.2–2.6 KB. KWS: tinymind 5.3 KB vs TFLM 5.7–6.3 KB. tinymind keeps
   every intermediate buffer static (no reuse); TFLM's arena planner reuses buffers but carries
   interpreter state. A ping-pong buffer pair (2,592 B + 784 B instead of 4,432 B) would save about 1 KB.
7. **Float:** TFLM float32 (27.9k) beats emlearn (40.5k) on the MLP; emlearn walks its weights
   column-strided, but its code is tiny. TFLM float KWS is 2.1M cycles because this CMSIS-NN build has its
   float kernels disabled and falls back to the reference float kernels.

## x86 host timing (tinymind only)

`make host-bench` builds `host/tm_host_bench.cpp` once per optimization level and x86 SIMD backend, runs
each pinned to one core, and writes `results/host_x86.txt`. It times the same generated models the
firmware runs: whole-model ns per inference for both networks, then ns per call for each KWS layer (each
layer out of line, as on the board), as the median of 41 timed repetitions. Each build must reproduce the
expected output hashes or `make` fails, so a backend that is fast but wrong cannot report a number.
Backends are cumulative (`avx512_vnni` also enables AVX2 and AVX-512F, as a build for that CPU would). A
backend the CPU lacks per `/proc/cpuinfo` is still compiled but is skipped, not run. Override the set with
`HOST_BACKENDS=...` and the levels with `HOST_LEVELS=...`.

This compares tinymind against itself, the scalar build against its x86 vector backends; there is no other
framework on the host yet. A TFLM x86 build is the planned next step; CMSIS-NN is Arm-only, so it will
run TFLM's reference kernels.

Committed run: i7-11850H, g++ 13.3, at `a7de803`, ns per inference:

| Backend | MLP -O2 | MLP -O3 | KWS -O2 | KWS -O3 |
|---|---:|---:|---:|---:|
| scalar | 361 | 368 | 15,345 | 14,657 |
| AVX2 | 336 | 303 | 14,779 | **13,321** |
| AVX-512F | 304 | 333 | 15,261 | 14,530 |
| AVX-512 VNNI | **191** | 227 | 15,185 | 14,770 |

* **The MLP gains from wide vectors; KWS barely does.** The MLP's dot products are 16–64 elements, so
  AVX-512 VNNI is 1.9× scalar at -O2. KWS is about 70% stem conv, whose dot products are 9 elements long
  (one 3×3 window, one input channel). Below 16 elements every x86 backend routes to the scalar code, so
  KWS speed is set by scalar code and every backend is within 10% of scalar.
* **AVX-512 needs `-mavx512vl`.** Without it GCC cannot use EVEX-encoded 256- and 128-bit instructions
  for loop tails, and the VNNI MLP is about 16% slower. The Makefile passes it.
* **Expect about ±5% from code layout.** Repeat runs of one binary agree within 1%, but the same library
  code built into a different harness binary moved KWS by about 5%. Treat differences smaller than that as
  noise.
* **Scalar max-pool at -O2 is 4.8× slower than at -O3** (373 vs 77 ns): GCC does not vectorize it at -O2
  with baseline x86-64 flags, but does once `-mavx2` is on. It is about 2% of KWS.

## Cortex-M0+: NUCLEO-L031K6

The same harness, backends and models on the smallest class of part: an STM32L031K6, Cortex-M0+ (ARMv6-M)
at 32 MHz with one flash wait state and prefetch on, 32 KB flash, 8 KB SRAM, no FPU, no DSP extension,
no hardware divide. `make BOARD=l031 LEVEL=O2|O3 run` → `results/raw_l031_<LEVEL>.txt`;
`report.py l031` → `results/REPORT_l031.md`, which also has the per-layer tables.

What differs from the F411 run:

* **Timing:** the M0+ has no DWT cycle counter. `board/board_l031k6.cpp` runs SysTick at the core clock
  over its full 24-bit range and counts wraps in its interrupt, so `board_cycles()` is a free-running 32-bit
  cycle count like CYCCNT. A wrap that has happened but not yet been counted is caught by reading the
  counter with interrupts masked and checking the pending SysTick bit. Checked on the board with a scratch
  firmware: a busy loop measures exactly 3 cycles per iteration over 1M and 10M iterations (spanning
  wraps), and with a 1024-cycle SysTick period, 2M back-to-back reads never go backwards across ~110k
  wraps. With the pending-bit check removed, 29,631 of them do. An empty `be_invoke` measures 44 cycles.
* **Test set:** 360 MLP samples are 23 KB as int8 and 92 KB as float, so the L031 images carry the first
  32 (`gen/subset.py`, applied at build time). Accuracy below is over those 32. The expected tinymind hash
  for that prefix is in `fw/backend.hpp` and checked by `make host-check`. KWS keeps all 8 samples.
* **What does not fit:** no TFLM image links into 32 KB. `make BOARD=l031 fit` links each one with the same
  flags against an unbounded memory map (`board/unbounded.ld`), so the table can say how much it would
  need. Their RAM column is static RAM (tensor arena included) without stack, so it is a floor.
* There are no `+SMLAD` images: ARMv6-M has no SMLAD, and the `ARM_DSP` gate refuses to build there.

### MLP 64-32-16-10, sklearn digits (32 test samples)

| Framework | O2 cycles | O3 cycles | O3 us @32MHz | flash O2 / O3 (B) | RAM (B) | acc | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 35,637 | 35,963 | 1,124 | 5,488 / 5,784 | 200 | 96.88% | yes |
| TFLM int8 (CMSIS-NN) | does not fit | does not fit | - | 39,160 / 43,444 | 1,872+ | - | - |
| TFLM int8 (reference) | does not fit | does not fit | - | 33,944 / 38,292 | 1,776+ | - | - |
| TFLM float32 | does not fit | does not fit | - | 45,700 / 49,984 | 1,392+ | - | - |
| emlearn float32 (only MLP path; software float) | 712,272 | 707,113 | 22,097 | 18,720 / 18,988 | 972 | 96.88% | n/a (float) |

### KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10

| Framework | O2 cycles | O3 cycles | O3 us @32MHz | flash O2 / O3 (B) | RAM (B) | int8 bit-exact |
|---|---:|---:|---:|---:|---:|---|
| tinymind int8 (scalar) | 818,262 | 746,112 | 23,316 | 6,016 / 8,452 | 5,488 | yes |
| TFLM int8 (CMSIS-NN) | does not fit | does not fit | - | 83,000 / 90,468 | 5,136+ | - |
| TFLM int8 (reference) | does not fit | does not fit | - | 51,072 / 57,052 | 5,072+ | - |
| TFLM float32 | does not fit | does not fit | - | 82,852 / 90,304 | 14,448+ | - |

### Random forest, 10 trees, depth 6, sklearn digits, integer features (32 test samples)

emlearn has no int8 MLP: its float net is its only working MLP path (`eml_net_forward_q16`, behind
`use_fixedpoint=True`, is unimplemented stubs in 0.23.2). Trees are the model both libraries run in pure
integer arithmetic, so `gen/make_trees.py` trains one scikit-learn `RandomForestClassifier` (majority vote)
and emits it for both. tinymind gets it two ways, plus a vote loop (`qtree.hpp` has no forest helper): one
`tinymind::QDecisionTree` per tree walking a `QTreeNode<int8_t>` pool at run time (`tm_rf`), and the same
pool as `constexpr` data in one `tinymind::QStaticDecisionTree` per tree with the default `QTreeInlineEval`
policy (#223), which expands every node at compile time into a compare against an immediate threshold and a
branch (`tm_rf_inline`; `tm_rf_inline_u8` reads the same bytes as `uint8_t`), and with the
`QTreeCompactEval` policy, which walks a packed copy of the pool built at compile time: decision nodes only,
4 bytes each, in pre-order, and a deduplicated int8 leaf table (`tm_rf_compact`, `tm_rf_compact_u8`). emlearn runs through
`emlearn.convert()` with both of its methods: `inline` (one generated if/else function per tree, integer votes) and `loadable` (node data walked by
`eml_trees_predict`, which tallies votes in float). Features are 2 x the pixel value; see friction below for
why. Each backend's predictions hash-match its own host reference on the board.

| Framework | O2 cycles | O3 cycles | O3 us @32MHz | flash O2 / O3 (B) | RAM (B) | acc | matches own host reference |
|---|---:|---:|---:|---:|---:|---:|---|
| tinymind int8 (QDecisionTree x10 + vote) | 2,581 | 2,524 | 79 | 12,800 / 12,912 | 80 | 96.88% | yes |
| tinymind int8 (QStaticDecisionTree compact x10 + vote) | 1,705 | 1,649 | 52 | 2,984 / 3,080 | 72 | 96.88% | yes |
| tinymind uint8 (QStaticDecisionTree compact x10 + vote) | 1,705 | 1,649 | 52 | 2,984 / 3,080 | 72 | 96.88% | yes |
| tinymind int8 (QStaticDecisionTree inline x10 + vote) | 919 | 872 | 27 | 6,832 / 8,340 | 120 | 96.88% | yes |
| tinymind uint8 (QStaticDecisionTree inline x10 + vote) | 872 | 845 | 26 | 6,180 / 7,924 | 112 | 96.88% | yes |
| emlearn int16 (inline) | 917 | 841 | 26 | 6,032 / 8,256 | 184 | 96.88% | yes |
| emlearn int16 (loadable) | 6,267 | 6,409 | 200 | 7,148 / 7,140 | 336 | 96.88% | yes |

Cycles are the median over 33 inferences of different samples. A tree walk's length depends on the input,
so the min-max spread is 7-10% (26% for `loadable`, whose float vote tally also varies), against < 0.1% for
the networks.

Per layer at `-O3` (cycles): MLP fc0 25,625 / fc1 7,707 / fc2 2,830; KWS conv3x3 516,155, maxpool 24,921,
dwconv3x3 73,079, pwconv 137,483, avgpool 5,756, fc 2,830.

Findings:

1. **Only tinymind fits.** TFLM's runtime plus model alone needs 33.9 KB (reference kernels, MLP, `-O2`)
   to 90 KB, more than the part's 32 KB of flash before the harness or a single test input is added.
   tinymind needs 5.5 KB for the MLP and 6.0 KB for the KWS CNN, both at `-O2`. Thumb-1 code is larger
   than Thumb-2: TFLM's reference MLP grows from 25.7 KB on the M4 to 33.9 KB here (+32%); tinymind's MLP
   from 5.3 KB to 5.5 KB.
2. **Bit-exact on ARMv6-M.** Both tinymind int8 models hash-match the host and the F411 on the M0+, with no
   alignment faults on a core that has no unaligned access.
3. **MLP: int8 vs software float.** On a part without an FPU, tinymind's int8 MLP runs in 35.6k cycles
   (1.1 ms) and emlearn's float MLP in 712k (22 ms): 20x. That is not an int8-vs-int8 comparison, since
   emlearn has no int8 MLP; it is what each library offers for this network on this part. Both get 31/32 of
   the subset right.
4. **M0+ vs M4F, same code, cycles at `-O3`.** MLP 35,963 vs 20,486 (1.8x); KWS 746,112 vs 254,900
   (2.9x), against tinymind's scalar build on the F411. At 32 MHz the KWS CNN takes 23 ms.
   Requantization's 64-bit multiply is a libgcc call (`__aeabi_lmul`) on ARMv6-M, and the KWS stem is
   69% of the network; both are where M0+ tuning would start.
5. **Trees: tinymind's inline forest ties emlearn's.** The first run (#221) walked the forest:
   2,524 cycles at `-O3` to emlearn's generated if/else at 841 (3.0x), in 12.8 KB of flash to 6.0 KB. A
   walked `QTreeNode` costs about 20 Thumb-1 instructions (four loads, index arithmetic for the 12-byte node
   stride, the walk's cycle guard), and every leaf is a full 12-byte node, where an inline node is a load, a
   compare against an immediate and a branch. `QStaticDecisionTree` with `QTreeInlineEval` (#223) generates
   that same code from the same node pool at compile time, and the pool itself is not stored: 872 cycles at
   `-O3` (2.9x faster than the walk) and 919 at `-O2`, where emlearn takes 917. Read as `uint8_t`, the
   features load with Thumb-1's immediate-offset `LDRB` instead of a register-offset `LDRSB`, one
   instruction less per node: 872 at `-O2` (ahead of emlearn's 917) and 845 at `-O3` (emlearn 841, within
   0.5%), in 7.9 KB at `-O3` against emlearn's 8.3 KB and 6.2 KB at `-O2` against 6.0 KB. It also takes
   less RAM (112 B to 184 B). Where flash matters more than cycles, `QTreeCompactEval` packs the same
   pool at compile time into 4-byte decision nodes in pre-order (so the walk needs no cycle guard) and a
   deduplicated int8 leaf table: the whole forest is 3.0 KB at `-O2`, half of emlearn's smallest (6.0 KB)
   and a quarter of the 12-byte-node walk, at 1,705 / 1,649 cycles, 1.5x faster than that walk and about
   half the speed of inline. The compact walk reads each node's feature index at run time, so `uint8_t`
   features gain nothing there; both builds are the same code. Against emlearn's data-driven `loadable`
   method, the compact walk is 3.7-3.9x faster in 42% of the flash, and even the 12-byte-node walk is 2.5x
   faster, though that gap is mostly emlearn tallying votes in software float. Accuracy is identical on the subset; over
   all 360 test samples, emlearn's predictions agree with the forest on 358 (see friction).

## Integration friction observed

* **emlearn:** `pip install emlearn`, `emlearn.convert(model).save()`, include one header. Minutes.
  MLPs are float only; trees take `int16_t` features. Tree thresholds are cast to `int16_t` (truncated)
  and the split is `x < t`, where scikit-learn's is `x <= t`. On raw digits pixels, a threshold of 3.5
  becomes 3 and `x == 3` goes the wrong way: the forest above drops from 90.6% to 76.4% accuracy.
  Doubling the features makes every threshold an integer, which recovers it, but a test value that lands
  exactly on an even threshold still splits differently (2 of 360 test samples here). tinymind's
  `x <= floor(t)` is exact for any threshold.
* **TFLM:** `git clone` + `make ... microlite`, which downloads CMSIS, CMSIS-NN and flatbuffers. Converting
  needs full TensorFlow (~600 MB). Pitfalls hit here:
  - there is no pip `tflite-micro`
  - the schema header lives in `tensorflow/lite/micro/schema/`
  - the lib is built with `-funsigned-char`
  - a Keras `Flatten` with a dynamic batch adds SHAPE/STRIDED_SLICE/PACK ops unless `batch_size=1`
  - `arena_used_bytes()` under-reports the minimum arena by 100–300 B
  - the per-op profiler is compiled out of release builds (`TF_LITE_STRIP_ERROR_STRINGS`), so it needs a
    `release_with_logs` lib
* **tinymind:** header-only and has no `.tflite` importer. The TFLite→tinymind bridge in
  `gen/make_models.py` was written for this benchmark (tinymind ships PyTorch/ONNX importers in `apps/`).

## Reproduce

Needs `arm-none-eabi-gcc`, `openocd`, a NUCLEO-F411RE on USB, and Python 3 with `venv`.

```
cd benchmarks/framework_comparison_f411
make deps                      # .venv (tensorflow-cpu, emlearn, scikit-learn, numpy),
                               # TFLM clone at the pinned revision, six TFLM libs
make host-check                # tinymind int8 outputs match the expected hashes
make host-bench                # x86 host timing per SIMD backend -> results/host_x86.txt
make LEVEL=O2 run              # build + flash + capture -> results/raw_O2.txt
make LEVEL=O3 run              # -> results/raw_O3.txt
.venv/bin/python -I report.py  # -> results/REPORT.md

make BOARD=l031 tflm-libs      # TFLM libs for Cortex-M0+ (only `make fit` needs them)
make BOARD=l031 LEVEL=O2 run   # NUCLEO-L031K6 -> results/raw_l031_O2.txt (runs `make fit` too)
make BOARD=l031 LEVEL=O3 run
.venv/bin/python -I report.py l031  # -> results/REPORT_l031.md
```

`make models` regenerates `gen/out/` from scratch (needs the venv). `make probe` re-bisects the TFLM arena
sizes; update the `ARENA_*` values in the Makefile if they change.

With a single ST-LINK attached, `run` finds it. With several, the Makefile refuses to guess and asks for
`STLINK=<serial>` (`make stlink` lists the attached probes and their ports). Put `STLINK = <serial>` in an
untracked `local.mk` to stop repeating it, or one `STLINK_f411` / `STLINK_l031` per board. openocd and the serial capture both follow that serial, so
replugging, which renumbers `/dev/ttyACM*`, cannot point them at different boards.

## Layout

| Path | What |
|---|---|
| `board/` | Bare-metal F411 and L031 support: clock, USART2 on the ST-LINK VCP, cycle counter (DWT / SysTick), stack painting, linker scripts (`sections.ld` shared, `unbounded.ld` for `make fit`) |
| `fw/harness.cpp` | Times one backend and prints a `key=value` report |
| `fw/be_*.c*` | One backend per framework: tinymind, TFLM, emlearn, empty baseline |
| `gen/make_models.py` | Trains/quantizes the models and emits every framework's headers into `gen/out/` |
| `gen/make_trees.py` | Trains the random forest and emits it for tinymind and emlearn (`make models` runs it) |
| `gen/subset.py` | Cuts a test set down to its first N samples for the L031 build |
| `host/tm_check.cpp` | Host-side tinymind hash check |
| `host/tm_host_bench.cpp` | Host timing of tinymind, per x86 SIMD backend (`make host-bench`) |
| `report.py` | `results/raw_<LEVEL>.txt` → `results/REPORT.md`; `report.py l031` → `results/REPORT_l031.md` |
