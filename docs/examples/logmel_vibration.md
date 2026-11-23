---
title: Fixed-point Log-mel Front End
parent: Examples
nav_order: 63
layout: default
---

# Fixed-point Log-mel Front End

`cpp/qmel.hpp` computes log-mel spectrograms and MFCCs from int16 frames with integer arithmetic only. It is the front end that turns audio or vibration into features a small int8 network can classify. This example runs it over a synthetic 12-second vibration recording and compares every frame with the float definition.

## How it works

- The recording, sampled at 3,200 Hz, is a 29.5 Hz motor with harmonics and noise. From 6 seconds on, a bearing fault rings a 900 Hz resonance at 107 impacts per second. The level rises from -50 dBFS to -6 dBFS across the run.
- `QLogMelSpectrogram<256, 32>` windows each 256-sample frame (hop 128) with a Q1.15 Hann window. Pre-emphasis and windowing are exact in 64 bits. The frame is then shifted so its largest sample lies in [2^14, 2^15), which means a quiet frame reaches the int16 `QRFFT1D` at full scale.
- 32 triangular HTK mel filters from 5 to 1,600 Hz are summed in 64 bits, and a fixed-point natural log follows. The frame's shift and the fixed scalings come back out as an exact integer offset of the base-2 log.
- The output is int16 in Q8 (steps of 1/256 nat). `QMfcc<32, 13>` applies an orthonormal DCT-II.
- The float definition is torchaudio's `MelSpectrogram` (power 2, HTK mel scale) followed by a natural log.

## Build and run

```bash
cd examples/logmel_vibration
make release
make run
make plot      # needs matplotlib in a venv
```

## Output

```
|log-mel error| in nats, by band level below the frame's loudest band:
   0-10 dB:   944 bands  mean 0.0012  p99 0.0036  max 0.0055
  10-20 dB:  1159 bands  mean 0.0017  p99 0.0057  max 0.0083
  20-30 dB:  1550 bands  mean 0.0052  p99 0.0215  max 0.0320
  30-40 dB:  5133 bands  mean 0.0160  p99 0.0669  max 0.1131
    40+ dB:   782 bands  mean 0.0326  p99 0.1423  max 0.2885
```

![Fixed-point vs float log-mel]({{ site.baseurl }}/assets/plots/logmel_vibration.png)

The top row shows the float and fixed-point spectrograms; the fault's 900 Hz resonance lights up bands 20 to 23 at 6 seconds. The bottom left shows their signed difference. It is no larger at the start of the run, at -50 dBFS, than at the end, at -6 dBFS: the block scale makes the error independent of the input level. The bottom right shows the error by how far a band sits below its frame's loudest band. Within 20 dB the two agree to 0.01 nats (0.04 dB). Further down, the error grows toward the rounding floor of the 16-bit FFT.

For a Cortex-M0+ at `-Os`, with N = 256 and 32 bands:

- the code is 1,710 bytes, including `QRFFT1D`;
- the deepest stack is about 264 bytes;
- the workspace is 1,028 bytes;
- the host-built tables are 1,644 bytes of flash, plus 832 for a 13-coefficient DCT.

[Source on GitHub](https://github.com/danmcleran/tinymind/tree/master/examples/logmel_vibration)
