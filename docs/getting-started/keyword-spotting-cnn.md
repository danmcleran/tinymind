---
title: Keyword Spotting CNN on a Cortex-M
layout: default
parent: Getting Started
nav_order: 4
---

# A Keyword-Spotting CNN for a Cortex-M

Keyword spotting (KWS) is the workload that put the "TinyML" moniker on the map: always-on wake-word detection running on a sub-dollar microcontroller from a 100 mAh battery. The [`examples/kws_cortex_m/`](https://github.com/danmcleran/tinymind/tree/master/examples/kws_cortex_m) example builds a KWS-style convolutional pipeline out of TinyMind's 2D layers, measures per-layer cycles and bytes with the new bench harness, and ships a portable port stub so the same pipeline can be moved onto a real MCU.

This tutorial walks through the pipeline architecture, the layer composition in C++, how to read the benchmark CSV, and what changes when you move from a host run to a Cortex-M target.

## Why Keyword Spotting?

KWS is a canonical TinyML workload because:

- **Data shape is 2D.** The front-end produces an MFCC (or log-mel) spectrogram: a time-by-frequency tile. A 1D pipeline over raw audio is feasible but wastes the rich 2D structure that small CNNs are good at.
- **Latency budget is tight.** A human expects the wake-word to trigger in about 100 ms. Running a CNN at 10 Hz means each inference has a ~10 ms budget on a Cortex-M4 at 80 MHz -- roughly 800 k cycles.
- **Flash and RAM are tiny.** The reference TinyMLPerf KWS target fits in ~50 KB flash and ~20 KB RAM.

The whole pipeline -- including TinyMind's training code -- fits easily inside those budgets.

## Pipeline Architecture

The example uses a MobileNet-style depthwise-separable block sandwiched between a small regular Conv2D front-end and a Global-Average-Pool + 1x1 dense classifier at the tail:

```
input [20 x 20 x 1]        (synthetic MFCC-like tile)
  -> Conv2D 3x3, 8 filters           -> [18 x 18 x 8]
  -> MaxPool2D 2x2                   -> [9  x 9  x 8]
  -> DepthwiseConv2D 3x3             -> [7  x 7  x 8]
  -> PointwiseConv2D 8 -> 16         -> [7  x 7  x 16]
  -> GlobalAvgPool2D                 -> [16]
  -> PointwiseConv2D (dense) 16 -> 10 -> [10]  (class logits)
```

Why this shape:

- **Regular `Conv2D` first.** The first convolution extracts generic edge/frequency-band features and is cheap enough that full cross-channel mixing is worth it.
- **MaxPool halves the spatial dim.** Keeps the activation volume small for the next stage.
- **Depthwise-separable block.** `DepthwiseConv2D` + `PointwiseConv2D` together replace a full `Conv2D 8->16` block at roughly 1/8 the MACs for K=3.
- **Global Average Pool + 1x1 dense.** GAP collapses the 7x7x16 feature map to a 16-vector. A `PointwiseConv2D<..., 1, 1, 16, 10>` then acts as the final dense classifier. This combination replaces the big flatten-to-dense matrix that usually dominates flash on small CNNs.

All sizes are compile-time template parameters. Change the `using` aliases at the top of `kws_cortex_m.cpp` and the rest of the pipeline follows automatically.

## Declaring the Layer Types

Here is the layer type chain, directly from the example:

```cpp
using Value      = float;
using Conv1Type  = tinymind::Conv2D<Value, 20, 20, 1, 3, 3, 1, 1, 8>;
using Pool1Type  = tinymind::MaxPool2D<Value,
                                        Conv1Type::OutputHeight,
                                        Conv1Type::OutputWidth,
                                        8, 2, 2, 2, 2>;
using DwType     = tinymind::DepthwiseConv2D<Value,
                                              Pool1Type::OutputHeight,
                                              Pool1Type::OutputWidth,
                                              8, 3, 3, 1, 1>;
using PwType     = tinymind::PointwiseConv2D<Value,
                                              DwType::OutputHeight,
                                              DwType::OutputWidth,
                                              8, 16>;
using GapType    = tinymind::GlobalAvgPool2D<Value,
                                              PwType::OutputHeight,
                                              PwType::OutputWidth,
                                              16>;
using DenseType  = tinymind::PointwiseConv2D<Value, 1, 1, 16, 10>;
```

Each layer's output dimensions feed the next layer's input dimensions through compile-time constants like `Conv1Type::OutputHeight`. If you tweak the input size from 20x20 to 40x49 (a real MFCC tile), the rest of the chain recomputes itself.

## Static Allocation

Every buffer is statically allocated -- no heap, no `new`:

```cpp
Conv1Type  gConv1;
Pool1Type  gPool1;
DwType     gDw;
PwType     gPw;
GapType    gGap;
DenseType  gDense;

Value gInput[20 * 20 * 1];
Value gBufConv1[Conv1Type::OutputSize];
Value gBufPool1[Pool1Type::OutputSize];
Value gBufDw[DwType::OutputSize];
Value gBufPw[PwType::OutputSize];
Value gBufGap[GapType::OutputSize];
Value gBufDense[DenseType::OutputSize];
```

On an MCU, these land in `.bss` at link time. There's no malloc, no RTOS dependency, and no possibility of a late-night out-of-memory in the field.

## Forward Pass with Per-Layer Timing

The bench harness wraps each `forward()` call with a cycle counter read. The exact sequence:

```cpp
tinymind::bench::enableCycleCounter();        // one-time DWT init on Cortex-M
tinymind::bench::writeHeader(std::cout);

const auto t0 = tinymind::bench::readCycleCounter();
gConv1.forward(gInput, gBufConv1);
const auto t1 = tinymind::bench::readCycleCounter();
gPool1.forward(gBufConv1, gBufPool1);
const auto t2 = tinymind::bench::readCycleCounter();
gDw.forward(gBufPool1, gBufDw);
const auto t3 = tinymind::bench::readCycleCounter();
// ...continues through gPw, gGap, gDense

tinymind::bench::writeRow(std::cout,
    {"conv2d_3x3_8", sizeof(gConv1), sizeof(gBufConv1), t1 - t0});
// ...one row per layer
```

On a host build `readCycleCounter()` returns elapsed nanoseconds from `std::chrono::steady_clock`. On a Cortex-M target compiled with `-DTINYMIND_BENCH_CORTEX_M`, the same call reads `DWT->CYCCNT` for true hardware cycle counts.

## Building and Running

```bash
cd examples/kws_cortex_m
make release
make run
```

Sample output on a typical x86 host (units shown as "cycles" are nanoseconds on the host):

```
name,weight_bytes,activation_bytes,cycles
conv2d_3x3_8,640,10368,28185
maxpool2d_2x2,5184,2592,716
dwconv2d_3x3,640,1568,1018
pwconv2d_8x16,1152,3136,1788
global_avgpool2d,0,64,61
dense_16x10,1360,40,79

Summary:
  total weight bytes     : 8976
  peak activation bytes  : 10368
  total inference cycles : 31847
```

The weights are random, so the predicted class is meaningless -- the point is the cycle profile and the footprint report.

## Reading the CSV

- **`weight_bytes`** includes both the trained weights and the gradient buffer used for on-device training. If you only need inference, roughly half this number disappears with a future inference-only layer variant.
- **`activation_bytes`** is the size of the output buffer. On a tight MCU build the activation buffers can overlap (layer N's input buffer is reusable as layer N+1's output once its values have been consumed) -- only the **peak** activation needs to be provisioned. For this pipeline the peak is the first conv's 10 KB output.
- **`cycles`** on host is ns; on Cortex-M it's hardware cycles from DWT.

On a Cortex-M4 at 80 MHz, 800 k cycles = 10 ms of wall time. For this pipeline size, a reasonable target is well under that budget even in pure C++.

## Footprint Summary

| Component | Bytes (float32) |
|---|---|
| All layer weights + gradients | 8,976 |
| Peak activation buffer | 10,368 |
| Total static allocation | ~19 KB |

That fits in the RAM of a Cortex-M4 the size of an STM32L4 (64-128 KB) with an order of magnitude to spare. Swapping the value type to Q8.8 would quarter the weight term and halve the activation term; the `MaxPool2D` argmax array (used for backprop) is the next-largest term and is value-type-independent.

## Porting to a Real Cortex-M

The example ships a vendor-neutral [`port_stub.hpp`](https://github.com/danmcleran/tinymind/blob/master/examples/kws_cortex_m/port_stub.hpp) with three functions to fill in:

```cpp
namespace kws_port {
    bool readMicSample(int16_t& sample);  // microphone front-end
    void putChar(char c);                 // UART TX (for CSV output)
    void platformInit();                  // DWT init, UART baud, etc.
}
```

Plus a minimal `UartSink` that implements the `operator<<` overloads the bench harness needs, without pulling in `<iostream>`:

```cpp
struct UartSink {
    UartSink& operator<<(const char* s);
    UartSink& operator<<(size_t v);
    UartSink& operator<<(uint32_t v);
};
```

Build with `-DTINYMIND_BENCH_CORTEX_M` and point the bench harness at a `UartSink` instead of `std::cout`. The model pipeline itself is unchanged.

Replacing the synthetic input with a real MFCC extractor is a short hop: TinyMind already ships [`cpp/fft1d.hpp`]({{ site.baseurl }}/architectures/fft) with sin/cos tables for fixed-point FFTs -- that's the expensive part of MFCC. The remaining mel-filterbank + log step is straightforward to add.

## Next Steps

- **Train a model with real data.** Google's [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) dataset is the canonical KWS training set. Train in PyTorch, export weights via the pattern in [`examples/pytorch/`]({{ site.baseurl }}/training/pytorch-interop), and load via `setFilterWeight` / `setChannelWeight`.
- **Resize the model.** A real KWS CNN uses a ~40x49 MFCC tile and 4-5 depthwise-separable blocks. Swap the `using` aliases and the compile-time constants do the rest.
- **Add activation + batch norm.** The raw layers in this tutorial don't include an activation function or batch norm between them -- they are there to measure the linear-algebra cost. For a real model insert a ReLU (an element-wise loop over each activation buffer) and a `BatchNorm2D` (not yet shipped -- same pattern as `BatchNorm1D`) between blocks.
- **Try Q-format.** The same pipeline works unchanged with `typedef tinymind::QValue<16, 16, true> Value;`. On a part without an FPU you'll get a large speedup, and the activation memory halves.
