# tinymind's FFTs on the board

Cycles per forward transform for tinymind's FFTs on real input, measured on two Cortex-M boards:

* `QFFT1D<N>`, the int16 complex radix-2 FFT, fed the samples in its real buffer and zeros in its imaginary
  buffer;
* `QRFFT1D<N>`, the int16 real-input FFT, which returns the N/2 + 1 distinct bins at about half the work;
* `FFT1D<Q8.8, N>`, the `QValue` FFT with the Q8.8 type of the fixed-point example in the docs (`IMG=fft1d`).

`QFFT1D` and `QRFFT1D` use the same Q1.15 twiddle table from `buildQFFTTwiddles(N, ...)`. The library side is
described in [`docs/architectures/fft.md`](../../docs/architectures/fft.md).

## Boards and method

* **NUCLEO-F411RE**: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on.
  Cycles from the DWT cycle counter.
* **NUCLEO-L031K6**: STM32L031K6, Cortex-M0+ at 32 MHz, 1 flash wait state. The M0+ has no DWT cycle
  counter; the board code extends SysTick to 32 bits. N stops at 512 here: three 1,024-entry int16 buffers
  would take 6 KB of the 8 KB of RAM.
* arm-none-eabi-gcc 13.2.1, `-std=c++14`, at `-O2` and `-O3`. The board support (clock, UART, cycle counter,
  linker scripts) and the ST-LINK selection are shared with
  [`../framework_comparison_f411`](../framework_comparison_f411).
* Input: pseudo-random samples, a different frame (one of 8) on each run; full-scale int16 for `QFFT1D` and
  `QRFFT1D`, a quarter of that range for `FFT1D` (|x| < 32 in Q8.8). Each figure is the median of 33 runs after
  one untimed warm-up. The minimum and maximum agree with the median to within 0.05%, so the timing does not
  depend on the input.
* Each transform sits in its own `noinline` function and is timed alone. The copy of the samples into
  `QFFT1D`'s real buffer and the zeroing of its imaginary buffer are timed separately (`load_med`) and are
  not in the `QFFT1D` column; they would add 0.9 to 4.4% to it.
* The twiddle tables sit in flash, where a `const` table goes. `TW=ram` puts them in RAM instead. `FFT1D`
  copies its twiddles into the object, so its tables are always in RAM.
* Exactness: a host program (`gen_tw.cpp`) builds the twiddle tables and hashes every transform's output over
  the 8 frames. The board recomputes the hashes, and `make run` fails unless every one matches, so these
  timings are of transforms that produce the host's output bit for bit. Every result below matched.

## Results

tinymind at #285 (`FFT1D`) and #284 (`QFFT1D`, `QRFFT1D`). Cycles per forward transform.

### NUCLEO-F411RE (Cortex-M4F, 100 MHz)

| N | `QFFT1D` -O3 | `QRFFT1D` -O3 | Speedup -O3 | `QFFT1D` -O2 | `QRFFT1D` -O2 | Speedup -O2 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 10,136 | 4,685 | 2.16× | 9,507 | 4,534 | 2.10× |
| 128 | 22,757 | 10,808 | 2.11× | 21,550 | 10,327 | 2.09× |
| 256 | 50,165 | 24,103 | 2.08× | 47,585 | 23,002 | 2.07× |
| 512 | 112,346 | 53,427 | 2.10× | 105,805 | 51,036 | 2.07× |
| 1024 | 233,516 | 118,027 | 1.98× | 232,323 | 112,036 | 2.07× |

At 100 MHz and `-O3`, `QRFFT1D<256>` takes 0.24 ms and `QFFT1D<256>` 0.50 ms.

### NUCLEO-L031K6 (Cortex-M0+, 32 MHz)

| N | `QFFT1D` -O3 | `QRFFT1D` -O3 | Speedup -O3 | `QFFT1D` -O2 | `QRFFT1D` -O2 | Speedup -O2 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 24,575 | 11,657 | 2.11× | 23,709 | 11,253 | 2.11× |
| 128 | 55,290 | 27,013 | 2.05× | 53,534 | 25,496 | 2.10× |
| 256 | 124,247 | 60,255 | 2.06× | 120,048 | 56,916 | 2.11× |
| 512 | 269,153 | 133,729 | 2.01× | 266,241 | 126,876 | 2.10× |

At 32 MHz and `-O3`, `QRFFT1D<256>` takes 1.9 ms and `QFFT1D<256>` 3.9 ms.

### Twiddle tables in RAM (`TW=ram`, -O3)

| N | F411 `QFFT1D` | F411 `QRFFT1D` | Speedup | L031 `QFFT1D` | L031 `QRFFT1D` | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 9,790 | 4,388 | 2.23× | 24,248 | 11,433 | 2.12× |
| 128 | 21,593 | 10,086 | 2.14× | 54,514 | 26,532 | 2.05× |
| 256 | 47,000 | 22,276 | 2.11× | 120,654 | 59,165 | 2.04× |
| 512 | 104,300 | 48,659 | 2.14× | 265,047 | 129,374 | 2.05× |
| 1024 | 212,324 | 107,238 | 1.98× | | | |

RAM tables save 3 to 9% on the F411 and 1 to 3% on the L031.

### `FFT1D<Q8.8>` (`IMG=fft1d`)

| N | F411 -O3 | F411 -O2 | L031 -O3 | L031 -O2 |
|---:|---:|---:|---:|---:|
| 64 | 15,047 | 14,844 | 39,340 | 39,833 |
| 128 | 33,787 | 33,336 | 89,790 | 90,911 |
| 256 | 74,548 | 73,936 | 202,696 | 204,691 |

`FFT1D<Q8.8, 256>` takes 6.3 ms on the 32 MHz M0+ at `-O3`. It is slower than `QFFT1D` at every size, on
both boards.

## What the numbers say

* `QRFFT1D` is 1.98 to 2.16 times faster than `QFFT1D` on the Cortex-M4F at `-O3`, and 2.01 to 2.11 times on
  the Cortex-M0+. Across both levels and both table placements the range is 1.98 to 2.23.
* The cost per radix-2 butterfly is smooth in N: on the F411 at `-O3`, 46 to 53 cycles for `QFFT1D`; on the
  L031, 117 to 128.
* `FFT1D<Q8.8>` costs 72 to 78 cycles per butterfly on the F411 at `-O3` and 198 to 205 on the L031, also
  smooth in N since #285 (below).

## What changed in #284

Before #284, `QFFT1D` and `QRFFT1D` cost 78 to 114 cycles per butterfly on the F411, and sizes with an even
number of stages (64, 256, 1,024) cost about 20% more per butterfly than 128 and 512 (up to 50% in some
builds), on both cores. Two
causes, both in GCC 13's code for the butterfly loop:

* The int16 saturation, written as early-return compares, compiled to two compare-and-branch pairs per value,
  four per butterfly. It is now one `SSAT` where the compiler reports the instruction (`__ARM_FEATURE_SAT`, as on
  the Cortex-M4) and a conditional elsewhere.
* The stage and group bookkeeping lived in registers across the inner loop, so the butterfly's own values
  were spilled to the stack: 16 stack accesses per butterfly in one build, 1 in another, and which one a
  length got changed with unrelated code. Each stage is now its own out-of-line function in which only three
  values live across the inner loop.

The outputs are unchanged: the host-side hashes for every transform and every N equal those of the earlier
code. Against the earlier results (the tables in this file's history):

* F411: `QFFT1D` 1.75 to 2.32 times faster at `-O3` and 1.67 to 2.31 times at `-O2`; `QRFFT1D` 1.64 to 2.00
  and 1.66 to 2.19 times.
* L031: `QFFT1D` 1.04 to 1.33 times faster at `-O3` and 1.03 to 1.57 times at `-O2`; `QRFFT1D` up to 1.29
  times faster at `-O3` and 1.10 to 1.15 times at `-O2`. Two L031 figures are slower: `QRFFT1D<128>` at `-O3`
  by 0.1% (27,013 against 26,987 cycles), and `QFFT1D<512>` with the tables in RAM by 5% (265,047 against
  252,922).

## What changed in #285

`FFT1D` had the same loop structure as `QFFT1D` before #284, and on the F411 at `-O3` its cost per butterfly
varied with N in the same way: 104, 81 and 99 cycles at N = 64, 128 and 256. Its stages are now
`detail::fftButterflyStage`, out of line, with the same three values across the inner loop; each butterfly's
arithmetic is written exactly as before. The outputs are unchanged: for Q8.8 (round-up and truncating),
Q16.16, float and double, forward and inverse, at N = 2 to 1,024, the new code matches the old bit for bit
on the host at `-O2` and at `-O3 -march=x86-64-v3`, and the board hashes equal the host's. Against the
earlier results, `FFT1D<Q8.8>` is 1.08 to 1.36 times faster on the F411 at `-O3`, 1.03 to 1.06 times at
`-O2`, 1.15 to 1.23 times on the L031 at `-O3` and 1.06 to 1.07 times at `-O2`.

## Reproduce

With `arm-none-eabi-gcc`, `openocd`, a host `g++`, and the board attached (probe serials in
`../framework_comparison_f411/local.mk`, as `STLINK_f411=` and `STLINK_l031=`, or `STLINK=<serial>`):

```
make LEVEL=O3 run                  # F411 -> results/raw_f411_O3.txt
make LEVEL=O2 run
make BOARD=l031 LEVEL=O3 run       # L031 -> results/raw_l031_O3.txt
make BOARD=l031 LEVEL=O2 run
make TW=ram LEVEL=O3 run           # -> results/raw_f411_O3_twram.txt
make BOARD=l031 TW=ram LEVEL=O3 run
make IMG=fft1d LEVEL=O3 run        # -> results/raw_fft1d_f411_O3.txt
make IMG=fft1d BOARD=l031 LEVEL=O2 run
```

`make` alone builds the image without flashing it. A changed tinymind header rebuilds the image and the
host-side hashes, so a run never re-measures a stale binary. `EXTRA=<flags>` adds compiler flags to the board
image.
