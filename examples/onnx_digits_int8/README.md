# onnx_digits_int8 (ONNX model -> generated tinymind model with a planned arena)

A small residual int8 CNN classifies the UCI optical digits (8x8 bitmaps,
1,797 test digits). Everything the C++ program runs is generated from an
ONNX model that ONNX Runtime quantized:

  * `weights.hpp` holds the int8 weights, int32 biases and requantizers
    (`apps/import_onnx/tinymind_import_onnx.py`);
  * `model.hpp` holds one `constexpr` tinymind layer object per layer, a
    static memory plan for every intermediate activation, and
    `forward(input, output, arena)` (`apps/import_onnx/tinymind_model_codegen.py`);
  * `model_int4.hpp` is the same model with its dense layer requantized to
    packed, per-channel int4 weights (`QDenseInt4`), in namespace
    `digits::int4`. It shares `weights.hpp`.
  * `model_int16x8.hpp` is the same model with int16 activations and its
    int8 weights (TFLite's 16x8 scheme), in namespace `digits::int16x8`. It
    shares `weights.hpp` too.
  * `reference.hpp` holds the input quantization tables; for each test
    digit, the class the float PyTorch model and ONNX Runtime's int8 model
    predict; and, for the first 256 digits, the logits of the int8-weight
    model computed with float activations.

`onnx_digits_int8.cpp` reads the test set, maps each pixel onto the
model's int8 input grid, calls `forward()`, and compares the result with
the labels and with ONNX Runtime. It is built with `TINYMIND_ENABLE_FLOAT=0`
and `TINYMIND_ENABLE_STD=0`, so the generated model uses neither.

## The model

```
input 8x8x1
  -> Conv 3x3 (8 filters) + ReLU                 c1_Conv
  -> Conv 3x3 (8) + ReLU, added to its input     c2_Conv, Add
  -> MaxPool 2x2                                 MaxPool
  -> depthwise Conv 3x3 + ReLU                   dw_Conv
  -> pointwise Conv 1x1 (16) + ReLU              pw_Conv
  -> flatten -> dense (10 logits)                fc_Gemm
```

Training uses PyTorch (Adam, cosine schedule, 60 epochs, seed 1) on the
3,823 training digits. ONNX Runtime's `quantize_static` then quantizes the
model in QDQ format with per-channel weights, calibrated on 512 training
digits.

## Results

```
test digits:                    1797
tinymind int8 accuracy:         0.9699
ONNX Runtime int8 accuracy:     0.9699
float (PyTorch) accuracy:       0.9699
same class as ONNX Runtime:     1795 of 1797
activation arena:               1024 bytes (unshared 2048, busiest layer 1024)
int4 dense layer accuracy:      0.9683 (same class as int8 on 1775 of 1797)
weights:                        3408 bytes int8, 2128 bytes with the dense layer in int4
16x8 accuracy:                  0.9688 (arena 2048 bytes, same weights)
same class as the float model:  int8 1791, 16x8 1795 of 1797
logit error vs float activations (first 256 digits): mean 0.1619 int8, 0.00128 16x8; max 0.6565 int8, 0.00633 16x8
```

The generated int8 model is as accurate as the float model. It predicts the
same class as ONNX Runtime for all but two digits. On both, ONNX Runtime's
two highest int8 logits are one step apart (69 and 70, 61 and 62), so a
requantization that the two runtimes round differently decides the class.
tinymind is right on one of the two digits and ONNX Runtime on the other.

The memory plan puts all seven layers' activations in 1,024 bytes. Giving
each output its own buffer would take 2,048 bytes. 1,024 bytes is also the
minimum: the first convolution's output and the second convolution's output
are both live while the residual `Add` runs, and each is 512 bytes. The `Add`
writes its result over the first convolution's output, which it is the last
layer to read.

![Memory plan and test confusion](../../docs/assets/plots/onnx_digits_int8.png)

At the top left, each box is one arena buffer. It spans the layers its
output is live for and sits at the bytes it occupies. At the top right is the
confusion matrix of the generated int8 model. The bottom panel is the
16x8 model's logit error next to the int8 model's, described below.

Compiled for a Cortex-M0+ (`arm-none-eabi-g++ -std=c++14 -mcpu=cortex-m0plus
-mthumb -Os -ffunction-sections -fdata-sections -c`), a translation unit
holding only `model.hpp`, static input, output and arena buffers, and a
`main` that calls `forward` is 7,326 bytes of code and read-only data
(kernels, weights, requantizers and layer objects). It has no `.data`, and
its 1,098 bytes of `.bss` are the arena plus the input and output buffers.
Nothing is allocated on the heap.

### int4 dense layer

The dense layer holds 2,560 of the model's 3,408 weight bytes.
`model_int4.hpp` requantizes it to symmetric int4, with one scale per
output channel, and stores two weights per byte:

| | int8 | int4 dense layer |
|---|---|---|
| Accuracy | 97.0% | 96.8% (3 more errors in 1,797) |
| Same class as the int8 model | | 1,775 of 1,797 digits |
| Weight bytes | 3,408 | 2,128 |
| Cortex-M0+ code and read-only data | 7,326 bytes | 5,988 bytes |

That is 18% less flash for the whole model. On the boards
([`benchmarks/tier1_onboard`](../../benchmarks/tier1_onboard)) the scalar
int4 kernel costs a little time: the whole model takes 2.6% longer on the
Cortex-M0+ (780k cycles against 761k at `-O3`) and 1% longer on the
Cortex-M4 with scalar kernels. With the Arm DSP backend the int4 dense
layer runs on SMLAD too, faster than the int8 one (7.6k cycles against
8.4k), and the model is 0.6% faster than int8 (170k cycles against 171k).
The requantization is
post-training, with no retraining; quantization-aware training would
recover more of the accuracy.

### int16 activations (16x8)

`model_int16x8.hpp` keeps the int8 weights and runs every activation in
int16, as TFLite's 16x8 scheme does. The code generator derives it from the
same int8 ONNX model with no recalibration. Each activation tensor gets a
symmetric int16 grid over the real range its int8 grid covered, 256 times
finer, and each layer clamps to that range, so a ReLU that the int8 model
folded into a zero point of -128 still clamps at zero. Biases become int64,
and each layer requantizes its int64 accumulator with TFLite's 64-bit
rounding.

Both quantized models approximate the same thing: the int8 weights run with
real-valued activations, each clipped to its calibrated range.
`reference.hpp` holds those logits for the first 256 test digits, computed
in float by `float_reference` in the code generator:

| | int8 | 16x8 |
|---|---|---|
| Mean logit error (logits span about 73) | 0.162 | 0.0013 (126x smaller) |
| Largest logit error | 0.657 | 0.0063 |
| Same class as the float PyTorch model | 1,791 of 1,797 | 1,795 of 1,797 |
| Accuracy | 97.0% | 96.9% (2 more errors in 1,797) |
| Activation arena | 1,024 bytes | 2,048 bytes |
| Weight bytes | 3,408 | 3,408 |
| Cortex-M0+ code and read-only data | 7,326 bytes | 9,638 bytes |
| Cortex-M0+ `.bss` | 1,098 bytes | 2,196 bytes |

For a classifier this small, int8 activations already cost nothing in
accuracy, so the 16x8 model is no more accurate. What 16-bit activations
buy is precision in the outputs: a regression target, a spectrogram feature
or a control signal that int8's 256 steps would quantize coarsely. The
price is twice the activation RAM, and 2,312 bytes more code: the 64-bit
requantization, a second copy of each layer's output loop for the fast
32-bit form, and the int16 add's loops. The weights do not change. The dot products stay in 32 bits,
summing blocks of 256 terms before widening. On the boards
([`benchmarks/tier1_onboard`](../../benchmarks/tier1_onboard), `-O3`):

| | int8 | 16x8 |
|---|---:|---:|
| Cortex-M0+ (NUCLEO-L031K6, 32 MHz) | 760,608 cycles (23.8 ms) | 853,668 cycles (26.7 ms) |
| Cortex-M4, scalar kernels (NUCLEO-F411RE, 100 MHz) | 479,524 (4.8 ms) | 452,807 (4.5 ms) |
| Cortex-M4, Arm DSP backend | 171,384 (1.7 ms) | 202,690 (2.0 ms) |

With the Arm DSP backend the 16x8 convolutions and dense layer use SMLAD on
int16 activation pairs, and each convolution output is requantized with one
32 x 32 -> 64 multiply-accumulate while its accumulator provably fits int32.
The int16 add rescales one input with a shift and the other with one SMLAL.
Each 16x8 load still carries two activations to int8's four: the 16x8 model
is 18% slower than int8 there, 12% slower on the Cortex-M0+, and 6% faster
without the DSP backend. The per-layer split is in the bench README.

## Build and run

```bash
cd examples/onnx_digits_int8
make release
make run      # accuracy, agreement with ONNX Runtime, CSVs in output/
make bench    # output/onnx_digits_int8_bench.csv: microseconds per inference on the host
make golden   # output/onnx_digits_int8.golden: int8 logits of the first 16 test digits
make plot     # output/onnx_digits_int8.png (needs matplotlib in a venv)
```

`make run` writes `output/onnx_digits_int8.csv` (one row per test digit:
label, the int8, int4 and 16x8 predictions, ONNX Runtime and float
predictions, and the 64 pixels), `output/onnx_digits_int8_plan.csv` (the
arena buffers), `output/onnx_digits_int8_summary.csv` (arena, unshared and
busiest-layer bytes, accuracies and logit errors) and
`output/onnx_digits_int8_logits.csv` (each of the first 256 digits' logits
from the float-activation reference, the int8 model and the 16x8 model).

## Regenerating the model

The generated headers are committed, so building the example needs no
Python. To retrain and regenerate them, use a virtual environment with
`torch`, `onnx`, `onnxruntime` and `numpy` (never the system Python):

```bash
python3 -m venv .venv && .venv/bin/pip install numpy torch onnx onnxruntime
make regenerate PY=.venv/bin/python
```
