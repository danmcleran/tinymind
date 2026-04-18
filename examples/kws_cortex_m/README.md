# KWS-style Cortex-M example

A host runner that exercises the new 2D layers (`Conv2D`,
`DepthwiseConv2D`, `PointwiseConv2D`, `MaxPool2D`, `GlobalAvgPool2D`)
plus the benchmark harness, arranged as a keyword-spotting-style
pipeline.

## Pipeline

```
input [20 x 20 x 1]
  -> Conv2D 3x3, 8 filters           -> [18 x 18 x 8]
  -> MaxPool2D 2x2                   -> [9  x 9  x 8]
  -> DepthwiseConv2D 3x3             -> [7  x 7  x 8]
  -> PointwiseConv2D 8 -> 16         -> [7  x 7  x 16]
  -> GlobalAvgPool2D                 -> [16]
  -> PointwiseConv2D (dense) 16 -> 10
```

Everything is statically allocated. No heap, no RTTI, no exceptions.

## Build and run

```bash
make release
make run
```

Sample output (CSV + summary):

```
name,weight_bytes,activation_bytes,cycles
conv2d_3x3_8,640,10368,...
maxpool2d_2x2,5184,2592,...
...
```

`weight_bytes` includes both weights and gradients (Conv2D stores them
together for on-device training). For inference-only MCU builds, a
future templated variant can drop the gradient array and roughly halve
the weight footprint.

## Porting to a Cortex-M target

1. **Cycle counter.** Compile with `-DTINYMIND_BENCH_CORTEX_M` so
   `bench::readCycleCounter()` reads `DWT->CYCCNT` instead of the
   host nanosecond fallback. Call `bench::enableCycleCounter()` once
   at startup.
2. **Output sink.** Replace `std::cout` with a UART wrapper. A minimal
   `UartSink` is provided in `port_stub.hpp`; wire the three functions
   to your HAL.
3. **Input front-end.** Replace the synthetic random input with your
   MFCC extractor. The FFT at `cpp/fft1d.hpp` plus the sin/cos tables
   provide the signal-processing pieces.
4. **Trained weights.** Load via `setFilterWeight` /
   `setChannelWeight`. See `examples/pytorch/` for a PyTorch export
   pattern.

## Why these dimensions?

The 20x20 input keeps the example compile-fast and readable. Real KWS
models typically use 40x49 MFCC tiles. All dimensions are compile-time
template parameters — swap them in the `using` aliases at the top of
`kws_cortex_m.cpp` and the rest of the pipeline follows.
