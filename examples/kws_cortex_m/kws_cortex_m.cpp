/**
* Copyright (c) 2026 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

// KWS-style pipeline demonstrating the new 2D layers plus the bench harness.
//
// Pipeline (intended to resemble a keyword-spotting CNN on an MFCC tile):
//
//   input [20 x 20 x 1]
//     -> Conv2D 3x3, 8 filters           -> [18 x 18 x 8]
//     -> MaxPool2D 2x2                   -> [9  x 9  x 8]
//     -> DepthwiseConv2D 3x3             -> [7  x 7  x 8]
//     -> PointwiseConv2D 8 -> 16         -> [7  x 7  x 16]
//     -> GlobalAvgPool2D                 -> [16]
//     -> PointwiseConv2D (dense) 16 -> 10 (logits)
//
// Weights, activations, and scratch are all statically allocated. No heap.
// Host runner uses synthetic input and random weights; the goal is to show
// footprint and per-layer cycle counts, not to classify anything useful.
//
// To port to a Cortex-M target:
//   1. Compile with -DTINYMIND_BENCH_CORTEX_M so bench::readCycleCounter()
//      reads DWT->CYCCNT instead of the host nanosecond counter.
//   2. Replace std::cout below with a UART sink that implements
//      operator<<(const char*) and operator<<(size_t / uint32_t).
//   3. Replace the synthetic input generation with your MFCC front-end.
//      This project already has cpp/fft1d.hpp + sin/cos tables; MFCC is a
//      short hop from there.
//   4. Load trained weights with setFilterWeight / setChannelWeight. A
//      Python exporter pattern lives in examples/pytorch/.

#include "conv2d.hpp"
#include "depthwiseconv2d.hpp"
#include "pointwiseconv2d.hpp"
#include "pool2d.hpp"
#include "bench/platform.hpp"
#include "bench/report.hpp"

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <random>

namespace {

// Model dimensions.
constexpr size_t kH = 20;
constexpr size_t kW = 20;
constexpr size_t kNumClasses = 10;

using Value = float;

using Conv1Type  = tinymind::Conv2D<Value, kH, kW, 1, 3, 3, 1, 1, 8>;
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
using DenseType  = tinymind::PointwiseConv2D<Value, 1, 1, 16, kNumClasses>;

Conv1Type  gConv1;
Pool1Type  gPool1;
DwType     gDw;
PwType     gPw;
GapType    gGap;
DenseType  gDense;

Value gInput[kH * kW * 1];
Value gBufConv1[Conv1Type::OutputSize];
Value gBufPool1[Pool1Type::OutputSize];
Value gBufDw[DwType::OutputSize];
Value gBufPw[PwType::OutputSize];
Value gBufGap[GapType::OutputSize];
Value gBufDense[DenseType::OutputSize];

template<typename Layer>
void fillRandomWeights(Layer& layer, std::mt19937& rng, float scale)
{
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < Layer::TotalWeights; ++i)
    {
        layer.setWeight(i, dist(rng));
    }
}

} // namespace

int main()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> inDist(-1.0f, 1.0f);
    for (size_t i = 0; i < kH * kW; ++i)
    {
        gInput[i] = inDist(rng);
    }

    fillRandomWeights(gConv1, rng, 0.1f);
    fillRandomWeights(gDw,    rng, 0.1f);
    fillRandomWeights(gPw,    rng, 0.1f);
    fillRandomWeights(gDense, rng, 0.1f);

    tinymind::bench::enableCycleCounter();
    tinymind::bench::writeHeader(std::cout);

    const tinymind::bench::Cycles t0 = tinymind::bench::readCycleCounter();
    gConv1.forward(gInput, gBufConv1);
    const tinymind::bench::Cycles t1 = tinymind::bench::readCycleCounter();
    gPool1.forward(gBufConv1, gBufPool1);
    const tinymind::bench::Cycles t2 = tinymind::bench::readCycleCounter();
    gDw.forward(gBufPool1, gBufDw);
    const tinymind::bench::Cycles t3 = tinymind::bench::readCycleCounter();
    gPw.forward(gBufDw, gBufPw);
    const tinymind::bench::Cycles t4 = tinymind::bench::readCycleCounter();
    gGap.forward(gBufPw, gBufGap);
    const tinymind::bench::Cycles t5 = tinymind::bench::readCycleCounter();
    gDense.forward(gBufGap, gBufDense);
    const tinymind::bench::Cycles t6 = tinymind::bench::readCycleCounter();

    tinymind::bench::writeRow(std::cout,
        {"conv2d_3x3_8",     sizeof(gConv1), sizeof(gBufConv1), t1 - t0});
    tinymind::bench::writeRow(std::cout,
        {"maxpool2d_2x2",    sizeof(gPool1), sizeof(gBufPool1), t2 - t1});
    tinymind::bench::writeRow(std::cout,
        {"dwconv2d_3x3",     sizeof(gDw),    sizeof(gBufDw),    t3 - t2});
    tinymind::bench::writeRow(std::cout,
        {"pwconv2d_8x16",    sizeof(gPw),    sizeof(gBufPw),    t4 - t3});
    tinymind::bench::writeRow(std::cout,
        {"global_avgpool2d", 0u,             sizeof(gBufGap),   t5 - t4});
    tinymind::bench::writeRow(std::cout,
        {"dense_16x10",      sizeof(gDense), sizeof(gBufDense), t6 - t5});

    const size_t totalWeightBytes =
        sizeof(gConv1) + sizeof(gPool1) + sizeof(gDw) +
        sizeof(gPw) + sizeof(gDense);
    const size_t peakActivationBytes = sizeof(gBufConv1); // first layer dominates here

    std::cout << "\nSummary:\n"
              << "  total weight bytes     : " << totalWeightBytes << "\n"
              << "  peak activation bytes  : " << peakActivationBytes << "\n"
              << "  total inference cycles : " << (t6 - t0) << "\n";

    std::cout << "\nLogits: ";
    size_t argmax = 0;
    for (size_t c = 0; c < kNumClasses; ++c)
    {
        if (gBufDense[c] > gBufDense[argmax]) argmax = c;
        std::cout << gBufDense[c] << " ";
    }
    std::cout << "\nPredicted class (untrained random weights): " << argmax << "\n";

    return 0;
}
