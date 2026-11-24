# The Tier 1 work on the board

Cycles per call, measured on two Cortex-M boards, for what #295 to #300 added:

* the generated digits model of [`examples/onnx_digits_int8`](../../examples/onnx_digits_int8) in its three
  variants: int8 (`model.hpp`), int8 with the dense layer on packed int4 weights (`model_int4.hpp`), and int16
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

tinymind at #304 (the 16x8 kernels below; every other figure is unchanged from #300, cycle for cycle).
Cycles per call; milliseconds at the board's clock in brackets.

### Digits model (`examples/onnx_digits_int8`)

| | F411 -O3 scalar | F411 -O3 DSP | F411 -O2 scalar | F411 -O2 DSP | L031 -O3 | L031 -O2 |
|---|---:|---:|---:|---:|---:|---:|
| int8 | 479,524 (4.80) | 171,384 (1.71) | 493,987 | 193,213 | 760,608 (23.8) | 799,322 (25.0) |
| int4 dense layer | 484,344 (4.84) | 186,613 (1.87) | 501,290 | 209,517 | 780,075 (24.4) | 818,340 (25.6) |
| 16x8 | 470,563 (4.71) | 220,380 (2.20) | 559,546 | 261,051 | 906,538 (28.3) | 1,079,017 (33.7) |
| 16x8 at #303 | 552,158 | 314,476 | 630,391 | 333,536 | 1,106,337 | 1,269,623 |
| 16x8 at #300 | 561,802 | 561,802 | 660,141 | 660,141 | 1,202,720 | 1,407,416 |
| int4 / int8 | 1.01 | 1.09 | 1.01 | 1.08 | 1.03 | 1.02 |
| 16x8 / int8 | 0.98 | 1.29 | 1.13 | 1.35 | 1.19 | 1.35 |

### Per layer, F411 -O3

| Layer (MACs) | int8 scalar | int8 DSP | int4 DSP | 16x8 scalar | 16x8 DSP | 16x8 DSP at #303 |
|---|---:|---:|---:|---:|---:|---:|
| c1_conv 3x3, 1 -> 8 (4,608) | 40,011 | 30,068 | 29,995 | 35,192 | 32,463 | 60,980 |
| c2_conv 3x3, 8 -> 8 (36,864) | 381,302 | 96,429 | 96,415 | 345,821 | 115,427 | 159,332 |
| add (512 elements) | 12,353 | 12,353 | 12,353 | 35,037 | 35,036 | 35,036 |
| maxpool 2x2 | 2,070 | 2,070 | 2,067 | 2,113 | 2,113 | 2,112 |
| dw_conv 3x3 (1,152) | 9,014 | 9,014 | 9,016 | 10,495 | 10,484 | 17,762 |
| pw_conv 1x1, 8 -> 16 (2,048) | 17,743 | 14,006 | 13,993 | 17,781 | 13,801 | 28,736 |
| fc 256 -> 10 (2,560) | 17,564 | 8,366 | 23,823 | 23,616 | 11,533 | 11,530 |

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
| digits int8 | 636 B | 1,588 B |
| digits int4 | 636 B | 1,524 B |
| digits 16x8 | 936 B | 1,460 B |
| log-mel frame | 88 B | 148 B |
| MFCC, k-means, Gaussian, learning | 0 to 36 B | 0 to 36 B |

## What the numbers say

* **int4 saves flash and costs little time.** The int4 dense layer is a scalar kernel. On the F411 with the
  DSP backend it runs at 23.8k cycles against 8.4k for the int8 layer with SMLAD, 2.85 times slower, which
  costs the whole model 9%. Without the DSP backend, and on the Cortex-M0+, which has no SMLAD, the int4 layer
  is 35% slower than the scalar int8 one and the model 1 to 3%. The flash it saves (1,280 weight bytes here)
  is the reason to use it; an SMLAD path that unpacks two nibbles into a halfword pair would close the gap
  on the M4.
* **16x8 costs 29% on the F411 with the DSP backend, 19% on the Cortex-M0+, and nothing on the F411 without
  it** (2% less than scalar int8: the 16x8 c2_conv, 346k cycles, beats the scalar int8 one, 381k). At #300
  these were 3.28 times, 58% and 17%. #303 added an SMLAD kernel for the int16 x int8 dot product (zero
  point 0, as TFLite's 16x8 scheme has; c2_conv 378k -> 159k cycles, fc 25k -> 12k), a one-rounding form of
  `QAdd`'s int16 input rescale (43.8k -> 35.0k) and 32-bit depthwise windows. #304 requantizes every 16x8
  convolution output with one 32 x 32 -> 64 multiply-accumulate (`detail::FastRequantizer64`, SMLAL on the
  M4) whenever the layer's bias and weights bound the accumulator inside int32, as they do here: the output
  is the high word of acc * M + 2^31 + zero_point * 2^32, with M the Q0.15 multiplier scaled by 2^(17 -
  shift), in place of the 64-bit clamp, multiply and shifts. With SMLAD that took c1_conv from 61.0k to
  32.5k cycles, c2_conv from 159k to 115k, dw_conv from 17.8k to 10.5k and pw_conv from 28.7k to 13.8k;
  on the Cortex-M0+, which has no 64-bit multiply, the same result comes from two 16 x 16 multiplies in
  32 bits, and the model went from 1.11M to 907k cycles. What remains is the add, 2.8 times the int8 one:
  each element still rescales both inputs in 64 bits and goes through the generic output requantizer.
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
