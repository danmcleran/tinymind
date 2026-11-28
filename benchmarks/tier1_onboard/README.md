# The Tier 1 work on the board

Cycles per call, measured on two Cortex-M boards, for what #295 to #300 added:

* the generated digits model of [`examples/onnx_digits_int8`](../../examples/onnx_digits_int8) in its three
  variants: int8 (`model.hpp`), int8 with every weight layer on packed int4 weights (`model_int4.hpp`), and int16
  activations with int8 weights, 16x8 (`model_int16x8.hpp`);
* a 256-sample log-mel frame with 32 bands (`QLogMelSpectrogram<256, 32>`, pre-emphasis 0.97) and its 13
  MFCCs (`QMfcc<32, 13>`), the configuration of `examples/vibration_anomaly`;
* anomaly scores over 32 int8 features: `QKMeansAnomaly<32, 8>` (weighted, with a model the host trained with
  `QKMeansAnomalyTrainer`), `QDiagGaussianAnomaly<32, 4>`, and one on-device learning step,
  `QKMeansAnomalyTrainer<32, 8>::learn`.

## Boards and method

* **NUCLEO-F411RE**: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on.
  Cycles from the DWT cycle counter. Built twice: with the library's defaults (scalar kernels) and with
  `DSP=1`, which sets `TINYMIND_ENABLE_SIMD_ARM_DSP=1` (SMLAD dot products for int8 x int8).
* **NUCLEO-L031K6**: STM32L031K6, Cortex-M0+ at 32 MHz, 1 flash wait state, 32 KB of flash and 8 KB of RAM. The
  board code extends SysTick to a 32-bit cycle counter.
* arm-none-eabi-gcc 13.2.1, `-std=c++14`, at `-O2` and `-O3`. The board support and the ST-LINK selection are
  shared with [`../framework_comparison_f411`](../framework_comparison_f411).
* One workload group per image (`IMG=int8 | int4 | 16x8 | features`), so that each fits the L031K6's flash.
  On the F411 the model images also time each layer on its own (`LAYERS=1`, too big for the L031K6): the
  seven layer objects of the generated model, called in `forward()`'s order, each in its own `noinline`
  function, with the final output checked against `forward()`'s.
* Inputs: the first 8 test digits of the optical digits set, mapped through the models' input tables; 8
  synthetic vibration frames (two tones and noise) whose level falls 6 dB per frame, so quiet frames are
  timed too; 8 feature vectors near the trained clusters. Each figure is the median of 33 calls cycling
  through the 8 inputs, after one untimed warm-up. Minimum and maximum are within 0.15% of the median for
  every workload except the learning step (2%), whose work depends on how far a sample sits from its
  nearest centroid.
* Each workload sits in its own `noinline` function and is timed alone.
* Exactness: a host program (`gen_tables.cpp`) builds the tables with the library's host builders, trains
  the k-means model, and hashes every workload's outputs over the 8 inputs (for learning, the trainer's
  centroids and counts after the timed calls). The board recomputes the hashes, and `make run` fails unless
  every one matches. Every result below matched.

## Results

The digits model's figures (int8, int4 and 16x8) were re-measured after the four-row SMLAD kernel for dense
layers, which the int8 model's fc layer now uses; the 16x8 figures did not move, except by 22 cycles on the
L031 at -O2. The log-mel and anomaly figures are unchanged from #300, cycle for cycle.
Cycles per call; milliseconds at the board's clock in brackets.

### Digits model (`examples/onnx_digits_int8`)

| | F411 -O3 scalar | F411 -O3 DSP | F411 -O2 scalar | F411 -O2 DSP | L031 -O3 | L031 -O2 |
|---|---:|---:|---:|---:|---:|---:|
| int8 | 477,010 (4.77) | 166,055 (1.66) | 490,431 | 189,958 | 755,739 (23.6) | 788,671 (24.6) |
| int4 | 454,288 (4.54) | 172,028 (1.72) | 466,757 | 193,590 | 778,803 (24.3) | 809,169 (25.3) |
| int4 dense layer only, #306 | 484,344 | 170,377 | 501,290 | 192,477 | 780,075 | 818,340 |
| 16x8 | 452,807 (4.53) | 202,690 (2.03) | 541,952 | 243,774 | 853,668 (26.7) | 1,021,008 (31.9) |
| 16x8 at #304 | 470,563 | 220,380 | 559,546 | 261,051 | 906,538 | 1,079,017 |
| 16x8 at #303 | 552,158 | 314,476 | 630,391 | 333,536 | 1,106,337 | 1,269,623 |
| 16x8 at #300 | 561,802 | 561,802 | 660,141 | 660,141 | 1,202,720 | 1,407,416 |
| int4 / int8 | 0.95 | 1.04 | 0.95 | 1.02 | 1.03 | 1.03 |
| 16x8 / int8 | 0.95 | 1.22 | 1.11 | 1.28 | 1.13 | 1.29 |

### Per layer, F411 -O3

| Layer (MACs) | int8 scalar | int8 DSP | int4 DSP | 16x8 scalar | 16x8 DSP | 16x8 DSP at #303 |
|---|---:|---:|---:|---:|---:|---:|
| c1_conv 3x3, 1 -> 8 (4,608) | 40,011 | 30,020 | 30,085 | 35,141 | 32,511 | 60,980 |
| c2_conv 3x3, 8 -> 8 (36,864) | 381,281 | 96,736 | 96,043 | 345,821 | 115,427 | 159,332 |
| add (512 elements) | 12,349 | 12,349 | 12,349 | 17,331 | 17,315 | 35,036 |
| maxpool 2x2 | 2,075 | 2,067 | 2,070 | 2,113 | 2,112 | 2,112 |
| dw_conv 3x3 (1,152) | 9,054 | 9,050 | 9,983 | 10,495 | 10,484 | 17,762 |
| pw_conv 1x1, 8 -> 16 (2,048) | 15,750 | 10,201 | 15,607 | 17,781 | 13,784 | 28,736 |
| fc 256 -> 10 (2,560) | 17,395 | 6,925 | 7,598 | 23,616 | 11,533 | 11,530 |

At #300, before any 16x8 DSP path, the model's 16x8 layers took 62,423, 378,204, 43,750, 2,113, 17,997, 30,523
and 24,844 cycles, with and without `DSP=1` alike.

### Log-mel front end and anomaly scores

| | F411 -O3 | F411 -O2 | L031 -O3 | L031 -O2 |
|---|---:|---:|---:|---:|
| Log-mel frame, 256 samples, 32 bands | 46,878 (0.47) | 47,564 | 185,172 (5.79) | 187,724 |
| MFCC, 32 bands -> 13 | 4,112 | 4,103 | 33,567 (1.05) | 33,561 |
| k-means score, 32 features x 8 clusters | 3,569 | 3,900 | 23,571 (0.74) | 23,685 |
| Gaussian score, 32 features x 4 components | 1,967 | 2,476 | 12,436 (0.39) | 13,331 |
| k-means learning step, 32 x 8 | 5,461 | 5,622 | 30,700 (0.96) | 31,093 |

These use no int8 x int8 dot product, so `DSP=1` leaves them unchanged (the DSP figures are identical).

### Stack

The deepest stack each workload adds, above what the harness itself uses at the same point (`-O3`):

| | F411 DSP | L031 |
|---|---:|---:|
| digits int8 | 636 B | 1,420 B |
| digits int4 | 1,340 B | 2,092 B |
| digits 16x8 | 936 B | 1,460 B |
| log-mel frame | 88 B | 148 B |
| MFCC, k-means, Gaussian, learning | 0 to 36 B | 0 to 36 B |

## What the numbers say

* **int4 halves the weights and costs almost no time; it costs stack.** Since #307 every weight layer of the
  int4 model is int4 (1,704 weight bytes against 3,408). Each convolution unpacks its weights onto the stack
  for the length of its call (`QConv2DInt4`, up to `TINYMIND_INT4_CONV_UNPACK_MAX_BYTES`, 1,024 by default)
  and runs the int8 kernels, so its time is the int8 layer's plus one pass over its weights; the depthwise
  layer unpacks one channel's taps at a time. The dense layer computes from its packed weights, on SMLAD
  with the DSP backend since #306 (7.6k cycles, 23.8k with the scalar kernel before): two masks spread a
  word of eight packed weights into four halfword pairs, and two rows share one widening of the input. The
  int8 fc layer now takes 6.9k cycles (8.4k before the four-row kernel), so with the DSP backend the int4
  model is 3.6% slower than int8; it is 3.1% slower on the Cortex-M0+ and 4.8% faster than int8 on the F411
  without the DSP backend (its c2_conv runs the same int8 kernel over the unpacked weights on the stack in
  346k cycles, against 381k over the int8 weights in flash). The price is the unpacked weights on the
  stack, 576 bytes for c2_conv: the deepest stack grows by 704 bytes on the F411 and 672 on the L031. A first version that
  computed every convolution from the packed nibbles directly was 38% slower than int8 with the DSP backend
  and 43% slower on the Cortex-M0+.
* **16x8 costs 22% on the F411 with the DSP backend, 13% on the Cortex-M0+, and nothing on the F411 without
  it** (5% less than scalar int8: the 16x8 c2_conv, 346k cycles, beats the scalar int8 one, 381k). At #300
  these were 3.28 times, 58% and 17%. #303 added an SMLAD kernel for the int16 x int8 dot product (zero
  point 0, as TFLite's 16x8 scheme has; c2_conv 378k -> 159k cycles, fc 25k -> 12k), a one-rounding form of
  `QAdd`'s int16 input rescale (43.8k -> 35.0k) and 32-bit depthwise windows. #304 requantizes every 16x8
  convolution output with one 32 x 32 -> 64 multiply-accumulate (`detail::FastRequantizer64`, SMLAL on the
  M4) whenever the layer's bias and weights bound the accumulator inside int32, as they do here: the output
  is the high word of acc * M + 2^31 + zero_point * 2^32, with M the Q0.15 multiplier scaled by 2^(17 -
  shift), in place of the 64-bit clamp, multiply and shifts. With SMLAD that took c1_conv from 61.0k to
  32.5k cycles, c2_conv from 159k to 115k, dw_conv from 17.8k to 10.5k and pw_conv from 28.7k to 13.8k;
  on the Cortex-M0+, which has no 64-bit multiply, the same result comes from two 16 x 16 multiplies in
  32 bits. #305 halved the add, 35.0k -> 17.3k cycles: `FastAddInput`'s SMLAL form now covers int16 at
  TFLite's left shift of 15 (`FastAddInputWide`), and for TFLite's int16 ADD the input with the larger
  scale, whose ratio is exactly 1/2, is rescaled with a shift alone, the other with one SMLAL, and the
  output with one more, clamped once (a fused ReLU included). The add is still 1.4 times the int8 one; the
  16x8 c2_conv is 1.19 times the int8 one with SMLAD, which reads four int8 activations per load to the
  16x8 kernel's two.
* **The audio front end fits a Cortex-M0+ in real time.** At 3.2 kHz a 256-sample frame lasts 80 ms; the
  log-mel frame and a k-means score take 6.5 ms of it at 32 MHz (8% of the CPU), and learning one frame
  takes 1 ms more. On the F411 the same work takes 0.5 ms.

## Reproduce it

```bash
cd benchmarks/tier1_onboard
make all-run                        # F411, -O3, scalar kernels: every image
make DSP=1 all-run                  # F411 with the Arm DSP backend
make BOARD=l031 LEVEL=O2 all-run    # Cortex-M0+
make BOARD=l031 IMG=features run    # one image
```

`make run` writes `results/raw_<img>_<board>_<level>[_dsp][_layers].txt`. The probe serials come from
`../framework_comparison_f411/local.mk` (`STLINK_f411`, `STLINK_l031`) or `STLINK=<serial>`.
