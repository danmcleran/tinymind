---
title: Streaming TFLite Model (int8 vibration)
parent: Examples
nav_order: 62
layout: default
---

# Streaming TFLite Model (int8 vibration)

A small int8 CNN watches a machine's vibration for a bearing fault, one 20 ms frame at a time. The model reads a window of 32 frames (0.64 s), so a device that reruns it every 20 ms recomputes 31 frames it has already computed. `apps/import_tflite` generates two models from the one `.tflite` file: `forward()` over the whole window, and `step()`, which takes one new frame and keeps, between calls, the frames each layer still needs. Both give, for every window of the test recording, the outputs TensorFlow Lite for Microcontrollers (TFLM) computes.

## How it works

- **Signal:** synthetic vibration at 3,200 samples per second.
  - A motor whose speed wanders between 28 and 32 Hz, with four harmonics and broadband noise.
  - Once the bearing is damaged, about 100 impacts per second, each ringing a resonance near 900 Hz.
  - Every 20 ms, 16 log band energies of 100 Hz each make one frame.
- **Model:** `Conv 3x1 (16)` → `depthwise 5x1` → `Conv 1x1 (16)` → `depthwise 9x1` → `Conv 1x1 (16)` → `depthwise 18x1` → `Conv 1x1 (8)` → `Dense (2)` → softmax, over a `[32][1][16]` input with time along the height.
  - Every convolution along time has stride 1 and no padding.
  - The receptive field is exactly the 32-frame window.
- **Streaming code generation:** with `streaming=True` (or `--streaming` on the command line), the code generator writes `step(frame, output, state, arena)` and `reset(state)`.
  - It rewrites each layer that slides a window of K frames along time to read K frames and write one.
  - Every tensor such a layer reads gets a history of its last K frames in the state, so the layer reads them as one contiguous tensor and runs the same kernel and requantizer as in `forward()`.
  - `step()` moves each history back one frame and writes the newest at its end.
  - Pools, a mean and a dense layer over time stream the same way. A layer that strides or pads along time is refused.
- **Driver:** `tflite_streaming_int8.cpp` feeds the 16-second test recording to `step()` and runs `forward()` over the window ending at each frame. It compares both with TFLM's outputs, which `train.py` recorded in `reference.hpp`, and exits with an error if any differs.

## Results

| | `forward()` | `step()` |
|---|---|---|
| input | 512 B (32 frames) | 16 B (one frame) |
| scratch arena | 896 B | 24 B |
| state kept between calls | none | 560 B |
| weights | 1,936 B | the same 1,936 B |
| time per 20 ms frame on the build host (x86-64, `-O3`) | about 10.7 µs | about 0.9 µs |

- **Bit-exact:** all 1,536 outputs of `step()` and of `forward()` equal TFLM's.
- **False alarms:** none on the 269 healthy windows.
- **Detection:** the alarm holds from 8.16 s on, when the fault is at 27% of full severity. A three-frame alarm at 6.04 s comes when the fault has barely started.
- **On the Cortex-M4 (NUCLEO-F411RE):** the same streaming code generation runs the MLPerf Tiny streaming wake word model 13 times faster than TFLM reruns its whole window. See [Benchmarks]({{ site.baseurl }}/benchmarks).

![Streaming TFLite model]({{ site.baseurl }}/assets/plots/tflite_streaming_int8.png)

## Build and run

```bash
cd examples/tflite_streaming_int8
make release
make run
make bench     # microseconds per step() and per forward() on the host
make golden    # step()'s int8 outputs for the first 16 full windows
make plot      # needs matplotlib in a venv
```

The `.tflite` model, the generated headers and the TFLM reference are committed. `make regenerate PY=<venv python>` retrains and regenerates them; the venv needs `numpy`, `tensorflow` and `tflite-micro`.
