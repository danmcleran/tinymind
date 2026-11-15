# QFFT1D vs QRFFT1D on the board

Cycles per forward transform for tinymind's two int16 FFTs on real input, measured on two Cortex-M boards:

* `QFFT1D<N>`, the complex radix-2 FFT, fed the samples in its real buffer and zeros in its imaginary buffer;
* `QRFFT1D<N>`, the real-input FFT, which returns the N/2 + 1 distinct bins at about half the work.

Both use the same Q1.15 twiddle table from `buildQFFTTwiddles(N, ...)`. The library side is described in
[`docs/architectures/fft.md`](../../docs/architectures/fft.md).

## Boards and method

* **NUCLEO-F411RE**: STM32F411RE, Cortex-M4F at 100 MHz, 3 flash wait states with the ART accelerator on.
  Cycles from the DWT cycle counter.
* **NUCLEO-L031K6**: STM32L031K6, Cortex-M0+ at 32 MHz, 1 flash wait state. The M0+ has no DWT cycle
  counter; the board code extends SysTick to 32 bits. N stops at 512 here: three 1,024-entry int16 buffers
  would take 6 KB of the 8 KB of RAM.
* arm-none-eabi-gcc 13.2.1, `-std=c++14`, at `-O2` and `-O3`. The board support (clock, UART, cycle counter,
  linker scripts) and the ST-LINK selection are shared with
  [`../framework_comparison_f411`](../framework_comparison_f411).
* Input: full-scale pseudo-random int16 samples, a different frame (one of 8) on each run. Each figure is the
  median of 33 runs after one untimed warm-up. The minimum and maximum agree with the median to within
  0.06% on both boards, so the timing does not depend on the input.
* Each transform sits in its own `noinline` function and is timed alone. The copy of the samples into
  `QFFT1D`'s real buffer and the zeroing of its imaginary buffer are timed separately (`load_med`) and are
  not in the `QFFT1D` column; they would add 0.6 to 2.6% to it.
* The twiddle tables sit in flash, where a `const` table goes. `TW=ram` puts them in RAM instead.
* Exactness: a host program (`gen_tw.cpp`) builds the twiddle tables and hashes both transforms' outputs over
  the 8 frames. The board recomputes the hashes, and `make run` fails unless every one matches, so these
  timings are of transforms that produce the host's output bit for bit. Every result below matched.

## Results

tinymind at `7336296`. Cycles per forward transform.

### NUCLEO-F411RE (Cortex-M4F, 100 MHz)

| N | `QFFT1D` -O3 | `QRFFT1D` -O3 | Speedup -O3 | `QFFT1D` -O2 | `QRFFT1D` -O2 | Speedup -O2 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 21,196 | 9,133 | 2.32× | 21,955 | 9,122 | 2.41× |
| 128 | 39,925 | 17,744 | 2.25× | 35,931 | 17,178 | 2.09× |
| 256 | 107,095 | 47,181 | 2.27× | 109,186 | 48,948 | 2.23× |
| 512 | 201,845 | 89,762 | 2.25× | 179,408 | 86,414 | 2.08× |
| 1024 | 542,882 | 236,210 | 2.30× | 493,315 | 245,298 | 2.01× |

At 100 MHz, `QRFFT1D<256>` takes 0.47 ms and `QFFT1D<256>` 1.07 ms.

### NUCLEO-L031K6 (Cortex-M0+, 32 MHz)

| N | `QFFT1D` -O3 | `QRFFT1D` -O3 | Speedup -O3 | `QFFT1D` -O2 | `QRFFT1D` -O2 | Speedup -O2 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 30,644 | 12,383 | 2.47× | 27,828 | 12,987 | 2.14× |
| 128 | 57,305 | 26,987 | 2.12× | 55,177 | 27,998 | 1.97× |
| 256 | 165,570 | 77,747 | 2.13× | 188,815 | 65,711 | 2.87× |
| 512 | 279,344 | 140,090 | 1.99× | 293,176 | 142,151 | 2.06× |

At 32 MHz, `QRFFT1D<256>` takes 2.4 ms (-O3) and `QFFT1D<256>` 5.2 ms.

### Twiddle tables in RAM (`TW=ram`, -O3)

| N | F411 `QFFT1D` | F411 `QRFFT1D` | Speedup | L031 `QFFT1D` | L031 `QRFFT1D` | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 20,729 | 8,748 | 2.37× | 29,951 | 12,255 | 2.44× |
| 128 | 36,572 | 16,780 | 2.18× | 58,655 | 27,287 | 2.15× |
| 256 | 105,650 | 45,590 | 2.32× | 152,395 | 69,797 | 2.18× |
| 512 | 180,781 | 85,448 | 2.12× | 252,922 | 143,453 | 1.76× |
| 1024 | 478,218 | 226,030 | 2.12× | | | |

On the F411, RAM tables save 1 to 12% on `QFFT1D` and 3 to 5% on `QRFFT1D`. On the L031 the effect has no
pattern: from 2.4% slower to 10% faster depending on N and the transform, with each transform slower at one
size. Neither board changes the conclusion.

## What the numbers say

* `QRFFT1D` is 2.25 to 2.32 times faster than `QFFT1D` on the Cortex-M4F at `-O3`, and 1.99 to 2.47 times on
  the Cortex-M0+. Across both levels and both table placements the range is 1.76 to 2.87.
* The ratio moves with N and with the optimization level because the two transforms are separate loops that
  the compiler schedules separately; at N = 256 on the M0+, `-O2` happens to be 14% slower than `-O3` on
  `QFFT1D` and 15% faster on `QRFFT1D`. Read the ratio as "a little over 2", not as a constant.
* The cost per butterfly is not smooth in N. On both cores, at both levels, and in both transforms, sizes
  with an even number of stages (64, 256, 1,024) cost more per radix-2 butterfly than 128 and 512: about
  20% more on the F411 at `-O3`, up to 50% more in other builds. It persists with the tables in RAM, and the
  timing does not depend on the input. The cause has not been found. The F411 spends 78 to 114 cycles per
  `QFFT1D` butterfly, which leaves room for a faster butterfly.

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
```

`make` alone builds the image without flashing it. A rebuilt tinymind header rebuilds the image and the
host-side hashes, so a run never re-measures a stale binary.
