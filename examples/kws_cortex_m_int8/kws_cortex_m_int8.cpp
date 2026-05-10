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

// int8 quantized variant of examples/kws_cortex_m. Same KWS-style pipeline,
// same input dimensions, but every layer is replaced with the parallel
// quantized layer family from cpp/q*.hpp:
//
//   input [20 x 20 x 1] int8
//     -> QConv2D 3x3, 8 filters           -> [18 x 18 x 8]  int8
//     -> QMaxPool2D 2x2                   -> [9  x 9  x 8]  int8
//     -> QDepthwiseConv2D 3x3 (per-chan)  -> [7  x 7  x 8]  int8
//     -> QPointwiseConv2D 8 -> 16         -> [7  x 7  x 16] int8
//     -> QGlobalAvgPool2D                 -> [16]           int8
//     -> QPointwiseConv2D (dense) 16 -> 10 (int8 logits)
//
// Weights, biases, requantizer tables, and activations are all statically
// allocated — no heap. The host runner generates synthetic float weights,
// calibrates them with the helpers in cpp/include/qcalibration.hpp, and
// emits a CSV cycles/byte report directly comparable to the float example.
//
// Build flags: TINYMIND_ENABLE_FLOAT=1 + TINYMIND_ENABLE_STD=1 are needed
// for calibration on the host. The deployable inference path (the forward()
// calls) does not require either; a real Cortex-M target would consume
// pre-calibrated weight/requantizer tables and compile with FLOAT=0,
// STD=0, QUANTIZATION=1.

#include "qaffine.hpp"
#include "qconv2d.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
#include "qpool2d.hpp"
#include "qcalibration.hpp"
#include "bench/platform.hpp"
#include "bench/report.hpp"

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <random>

namespace {

constexpr std::size_t kH = 20;
constexpr std::size_t kW = 20;
constexpr std::size_t kNumClasses = 10;

using Q = int8_t;
using A = int32_t;

constexpr int32_t kQmin = -127;
constexpr int32_t kQmax = 127;
constexpr int32_t kZeroPoint = 0;          // symmetric int8 activations
constexpr float   kActScale  = 1.0f / 127.0f; // activations cover [-1, 1]
constexpr float   kWScale    = 0.1f / 127.0f; // weights drawn from [-0.1, 0.1]

using QConv1Type = tinymind::QConv2D<Q, Q, A, Q, kH, kW, 1, 3, 3, 1, 1, 8>;
using QPool1Type = tinymind::QMaxPool2D<Q,
                                         QConv1Type::OutputHeight,
                                         QConv1Type::OutputWidth,
                                         8, 2, 2, 2, 2>;
using QDwType    = tinymind::QDepthwiseConv2D<Q, Q, A, Q,
                                               QPool1Type::OutputHeight,
                                               QPool1Type::OutputWidth,
                                               8, 3, 3, 1, 1>;
using QPwType    = tinymind::QPointwiseConv2D<Q, Q, A, Q,
                                               QDwType::OutputHeight,
                                               QDwType::OutputWidth,
                                               8, 16>;
using QGapType   = tinymind::QGlobalAvgPool2D<Q, A,
                                               QPwType::OutputHeight,
                                               QPwType::OutputWidth,
                                               16>;
using QDenseType = tinymind::QPointwiseConv2D<Q, Q, A, Q, 1, 1, 16, kNumClasses>;

QConv1Type  gConv1;
QPool1Type  gPool1;
QDwType     gDw;
QPwType     gPw;
QGapType    gGap;
QDenseType  gDense;

// Activation buffers.
Q gInput[kH * kW * 1];
Q gBufConv1[QConv1Type::OutputSize];
Q gBufPool1[QPool1Type::OutputSize];
Q gBufDw[QDwType::OutputSize];
Q gBufPw[QPwType::OutputSize];
Q gBufGap[QGapType::OutputSize];
Q gBufDense[QDenseType::OutputSize];

// Weight + bias tables (caller-owned; quantized layers are templates over
// pointer fields, not embedded weights).
Q gW_conv1[QConv1Type::TotalWeights];
A gB_conv1[QConv1Type::NumFilters];

Q gW_dw[QDwType::TotalWeights];
A gB_dw[QDwType::Channels];
tinymind::Requantizer<A, Q> gReqDw[QDwType::Channels];

Q gW_pw[QPwType::TotalWeights];
A gB_pw[QPwType::NumFilters];

Q gW_dense[QDenseType::TotalWeights];
A gB_dense[QDenseType::NumFilters];

template<std::size_t N>
void calibratePerTensorWeights(Q (&dst)[N], std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    float wf[N];
    for (std::size_t i = 0; i < N; ++i) wf[i] = dist(rng);
    tinymind::quantizeBuffer<Q>(wf, dst, N, kWScale, 0, kQmin, kQmax);
}

} // namespace

int main()
{
    std::mt19937 rng(42);

    // Quantize the synthetic float input.
    {
        std::uniform_real_distribution<float> inDist(-1.0f, 1.0f);
        for (std::size_t i = 0; i < kH * kW; ++i)
        {
            gInput[i] = tinymind::quantize<Q>(inDist(rng), kActScale,
                                              kZeroPoint, kQmin, kQmax);
        }
    }

    // QConv1: per-tensor symmetric weight scale.
    calibratePerTensorWeights(gW_conv1, rng);
    for (std::size_t i = 0; i < QConv1Type::NumFilters; ++i) gB_conv1[i] = 0;
    gConv1.weights = gW_conv1;
    gConv1.biases  = gB_conv1;
    gConv1.input_zero_point = static_cast<Q>(kZeroPoint);
    gConv1.requantizer = tinymind::buildRequantizer<Q>(
        kActScale, kWScale, kActScale, kZeroPoint, kQmin, kQmax);

    // QDepthwise: per-channel symmetric weight scale (TFLite mandate).
    {
        float wf[QDwType::TotalWeights];
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (std::size_t i = 0; i < QDwType::TotalWeights; ++i) wf[i] = dist(rng);

        float per_channel_scale[QDwType::Channels];
        tinymind::computePerChannelSymmetricScales(
            wf, QDwType::Channels, QDwType::WeightsPerChannel, kQmax,
            per_channel_scale);

        for (std::size_t c = 0; c < QDwType::Channels; ++c)
        {
            tinymind::quantizeBuffer<Q>(
                wf + c * QDwType::WeightsPerChannel,
                gW_dw + c * QDwType::WeightsPerChannel,
                QDwType::WeightsPerChannel,
                per_channel_scale[c], 0, kQmin, kQmax);
            gB_dw[c] = 0;
            gReqDw[c] = tinymind::buildRequantizer<Q>(
                kActScale, per_channel_scale[c], kActScale,
                kZeroPoint, kQmin, kQmax);
        }
    }
    gDw.weights = gW_dw;
    gDw.biases  = gB_dw;
    gDw.input_zero_point = static_cast<Q>(kZeroPoint);
    gDw.requantizers = gReqDw;

    // QPointwise (8 -> 16).
    calibratePerTensorWeights(gW_pw, rng);
    for (std::size_t i = 0; i < QPwType::NumFilters; ++i) gB_pw[i] = 0;
    gPw.weights = gW_pw;
    gPw.biases  = gB_pw;
    gPw.input_zero_point = static_cast<Q>(kZeroPoint);
    gPw.requantizer = tinymind::buildRequantizer<Q>(
        kActScale, kWScale, kActScale, kZeroPoint, kQmin, kQmax);

    // QGlobalAvgPool: same scale on input and output, just clamp range.
    gGap.qmin = static_cast<Q>(kQmin);
    gGap.qmax = static_cast<Q>(kQmax);

    // QDense (16 -> 10) implemented as 1x1 pointwise.
    calibratePerTensorWeights(gW_dense, rng);
    for (std::size_t i = 0; i < QDenseType::NumFilters; ++i) gB_dense[i] = 0;
    gDense.weights = gW_dense;
    gDense.biases  = gB_dense;
    gDense.input_zero_point = static_cast<Q>(kZeroPoint);
    gDense.requantizer = tinymind::buildRequantizer<Q>(
        kActScale, kWScale, kActScale, kZeroPoint, kQmin, kQmax);

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

    // Q layers hold only pointers + Requantizers in their struct, so the
    // real weight footprint is the external buffer sizes. Roll those into
    // the weight_bytes column to match the float runner's accounting.
    const std::size_t conv1Bytes = sizeof(gConv1)  + sizeof(gW_conv1) + sizeof(gB_conv1);
    const std::size_t dwBytes    = sizeof(gDw)     + sizeof(gW_dw)    + sizeof(gB_dw)    + sizeof(gReqDw);
    const std::size_t pwBytes    = sizeof(gPw)     + sizeof(gW_pw)    + sizeof(gB_pw);
    const std::size_t denseBytes = sizeof(gDense)  + sizeof(gW_dense) + sizeof(gB_dense);

    tinymind::bench::writeRow(std::cout,
        {"qconv2d_3x3_8",     conv1Bytes,    sizeof(gBufConv1), t1 - t0});
    tinymind::bench::writeRow(std::cout,
        {"qmaxpool2d_2x2",    sizeof(gPool1), sizeof(gBufPool1), t2 - t1});
    tinymind::bench::writeRow(std::cout,
        {"qdwconv2d_3x3",     dwBytes,        sizeof(gBufDw),    t3 - t2});
    tinymind::bench::writeRow(std::cout,
        {"qpwconv2d_8x16",    pwBytes,        sizeof(gBufPw),    t4 - t3});
    tinymind::bench::writeRow(std::cout,
        {"qglobal_avgpool2d", sizeof(gGap),   sizeof(gBufGap),   t5 - t4});
    tinymind::bench::writeRow(std::cout,
        {"qdense_16x10",      denseBytes,     sizeof(gBufDense), t6 - t5});

    const std::size_t totalWeightBytes =
        conv1Bytes + sizeof(gPool1) + dwBytes + pwBytes +
        sizeof(gGap) + denseBytes;
    const std::size_t peakActivationBytes = sizeof(gBufConv1);

    std::cout << "\nSummary:\n"
              << "  total weight bytes     : " << totalWeightBytes << "\n"
              << "  peak activation bytes  : " << peakActivationBytes << "\n"
              << "  total inference cycles : " << (t6 - t0) << "\n";

    std::cout << "\nLogits (int8): ";
    std::size_t argmax = 0;
    for (std::size_t c = 0; c < kNumClasses; ++c)
    {
        if (gBufDense[c] > gBufDense[argmax]) argmax = c;
        std::cout << static_cast<int>(gBufDense[c]) << " ";
    }
    std::cout << "\nPredicted class (untrained random weights): " << argmax << "\n";

    return 0;
}
