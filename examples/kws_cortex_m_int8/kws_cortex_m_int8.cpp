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
// quantizes them with the helpers in cpp/include/qcalibration.hpp, fits
// each layer's activation scale from a float reference forward pass over
// calibration inputs (RangeObserver), and emits a CSV cycles/byte report
// directly comparable to the float example, followed by the int8 logits
// against the float model's.
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
#include <cmath>
#include <iomanip>
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
constexpr float   kInScale   = 1.0f / 127.0f; // the input covers [-1, 1]
constexpr float   kWScale    = 0.1f / 127.0f; // weights drawn from [-0.1, 0.1]

// Calibration inputs drawn after the weights. Every activation scale below
// the input comes from the float model's observed range over these.
constexpr std::size_t kCalibrationSamples = 32;

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

// The float model: the same weights before quantization. Biases are 0.
float gInputF[kH * kW];
float gWf_conv1[QConv1Type::TotalWeights];
float gWf_dw[QDwType::TotalWeights];
float gWf_pw[QPwType::TotalWeights];
float gWf_dense[QDenseType::TotalWeights];

template<std::size_t N>
void calibratePerTensorWeights(Q (&dst)[N], float (&wf)[N], std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (std::size_t i = 0; i < N; ++i) wf[i] = dist(rng);
    tinymind::quantizeBuffer<Q>(wf, dst, N, kWScale, 0, kQmin, kQmax);
}

// Float reference forward pass, same layouts as the Q layers (NHWC
// activations; conv weights OHWI, depthwise [C][KH][KW], pointwise [F][C]).
struct FloatActivations
{
    float conv1[QConv1Type::OutputSize];
    float pool1[QPool1Type::OutputSize];
    float dw[QDwType::OutputSize];
    float pw[QPwType::OutputSize];
    float gap[QGapType::OutputSize];
    float dense[QDenseType::OutputSize];
};

void floatForward(const float* in, FloatActivations& a)
{
    const std::size_t oh1 = QConv1Type::OutputHeight;
    const std::size_t ow1 = QConv1Type::OutputWidth;
    const std::size_t f1  = QConv1Type::NumFilters;
    for (std::size_t y = 0; y < oh1; ++y)
    {
        for (std::size_t x = 0; x < ow1; ++x)
        {
            for (std::size_t f = 0; f < f1; ++f)
            {
                float acc = 0.0f;
                for (std::size_t ky = 0; ky < 3; ++ky)
                {
                    for (std::size_t kx = 0; kx < 3; ++kx)
                    {
                        acc += in[(y + ky) * kW + (x + kx)] * gWf_conv1[(f * 3 + ky) * 3 + kx];
                    }
                }
                a.conv1[(y * ow1 + x) * f1 + f] = acc;
            }
        }
    }

    const std::size_t oh2 = QPool1Type::OutputHeight;
    const std::size_t ow2 = QPool1Type::OutputWidth;
    const std::size_t c2  = f1;
    for (std::size_t y = 0; y < oh2; ++y)
    {
        for (std::size_t x = 0; x < ow2; ++x)
        {
            for (std::size_t c = 0; c < c2; ++c)
            {
                float m = a.conv1[((2 * y) * ow1 + 2 * x) * c2 + c];
                for (std::size_t py = 0; py < 2; ++py)
                {
                    for (std::size_t px = 0; px < 2; ++px)
                    {
                        const float v = a.conv1[((2 * y + py) * ow1 + 2 * x + px) * c2 + c];
                        if (v > m) m = v;
                    }
                }
                a.pool1[(y * ow2 + x) * c2 + c] = m;
            }
        }
    }

    const std::size_t oh3 = QDwType::OutputHeight;
    const std::size_t ow3 = QDwType::OutputWidth;
    for (std::size_t y = 0; y < oh3; ++y)
    {
        for (std::size_t x = 0; x < ow3; ++x)
        {
            for (std::size_t c = 0; c < c2; ++c)
            {
                float acc = 0.0f;
                for (std::size_t ky = 0; ky < 3; ++ky)
                {
                    for (std::size_t kx = 0; kx < 3; ++kx)
                    {
                        acc += a.pool1[((y + ky) * ow2 + x + kx) * c2 + c] * gWf_dw[(c * 3 + ky) * 3 + kx];
                    }
                }
                a.dw[(y * ow3 + x) * c2 + c] = acc;
            }
        }
    }

    const std::size_t f4 = QPwType::NumFilters;
    for (std::size_t p = 0; p < oh3 * ow3; ++p)
    {
        for (std::size_t f = 0; f < f4; ++f)
        {
            float acc = 0.0f;
            for (std::size_t c = 0; c < c2; ++c)
            {
                acc += a.dw[p * c2 + c] * gWf_pw[f * c2 + c];
            }
            a.pw[p * f4 + f] = acc;
        }
    }

    for (std::size_t f = 0; f < f4; ++f)
    {
        float sum = 0.0f;
        for (std::size_t p = 0; p < oh3 * ow3; ++p)
        {
            sum += a.pw[p * f4 + f];
        }
        a.gap[f] = sum / static_cast<float>(oh3 * ow3);
    }

    for (std::size_t k = 0; k < kNumClasses; ++k)
    {
        float acc = 0.0f;
        for (std::size_t f = 0; f < f4; ++f)
        {
            acc += a.gap[f] * gWf_dense[k * f4 + f];
        }
        a.dense[k] = acc;
    }
}

// Symmetric int8 scale covering the observed range.
float activationScale(const tinymind::RangeObserver& obs)
{
    return tinymind::computeAffineParamsSymmetric(obs.min_value, obs.max_value, kQmax).scale;
}

std::size_t argmaxOf(const float* v, std::size_t n)
{
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
    {
        if (v[i] > v[best]) best = i;
    }
    return best;
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
            gInputF[i] = inDist(rng);
            gInput[i] = tinymind::quantize<Q>(gInputF[i], kInScale,
                                              kZeroPoint, kQmin, kQmax);
        }
    }

    // Weights. QConv1, QPointwise and QDense use one symmetric weight scale
    // per tensor; QDepthwise uses one per channel (TFLite mandate).
    calibratePerTensorWeights(gW_conv1, gWf_conv1, rng);

    float per_channel_scale[QDwType::Channels];
    {
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (std::size_t i = 0; i < QDwType::TotalWeights; ++i) gWf_dw[i] = dist(rng);

        tinymind::computePerChannelSymmetricScales(
            gWf_dw, QDwType::Channels, QDwType::WeightsPerChannel, kQmax,
            per_channel_scale);

        for (std::size_t c = 0; c < QDwType::Channels; ++c)
        {
            tinymind::quantizeBuffer<Q>(
                gWf_dw + c * QDwType::WeightsPerChannel,
                gW_dw + c * QDwType::WeightsPerChannel,
                QDwType::WeightsPerChannel,
                per_channel_scale[c], 0, kQmin, kQmax);
        }
    }

    calibratePerTensorWeights(gW_pw, gWf_pw, rng);
    calibratePerTensorWeights(gW_dense, gWf_dense, rng);

    // Activation scales. Fixed scales do not work here: with weights in
    // [-0.1, 0.1] each layer shrinks the signal several times over, so one
    // scale for every layer leaves the logits all 0. Instead, run the float
    // model over calibration inputs and give each layer's output the
    // symmetric scale that covers its observed range. Max pooling and
    // global average pooling have no requantizer, so their outputs keep
    // their input's scale.
    tinymind::RangeObserver obsConv1;
    tinymind::RangeObserver obsDw;
    tinymind::RangeObserver obsPw;
    tinymind::RangeObserver obsDense;
    {
        std::uniform_real_distribution<float> calDist(-1.0f, 1.0f);
        static float calInput[kH * kW];
        static FloatActivations calAct;
        for (std::size_t s = 0; s < kCalibrationSamples; ++s)
        {
            for (std::size_t i = 0; i < kH * kW; ++i) calInput[i] = calDist(rng);
            floatForward(calInput, calAct);
            obsConv1.observe(calAct.conv1, QConv1Type::OutputSize);
            obsDw.observe(calAct.dw, QDwType::OutputSize);
            obsPw.observe(calAct.pw, QPwType::OutputSize);
            obsDense.observe(calAct.dense, QDenseType::OutputSize);
        }
    }
    const float conv1Scale = activationScale(obsConv1);
    const float dwScale    = activationScale(obsDw);
    const float pwScale    = activationScale(obsPw);
    const float denseScale = activationScale(obsDense);

    // QConv1.
    for (std::size_t i = 0; i < QConv1Type::NumFilters; ++i) gB_conv1[i] = 0;
    gConv1.weights = gW_conv1;
    gConv1.biases  = gB_conv1;
    gConv1.input_zero_point = static_cast<Q>(kZeroPoint);
    gConv1.requantizer = tinymind::buildRequantizer<Q>(
        kInScale, kWScale, conv1Scale, kZeroPoint, kQmin, kQmax);

    // QDepthwise: its input is the max pool's output, at conv1's scale.
    for (std::size_t c = 0; c < QDwType::Channels; ++c)
    {
        gB_dw[c] = 0;
        gReqDw[c] = tinymind::buildRequantizer<Q>(
            conv1Scale, per_channel_scale[c], dwScale,
            kZeroPoint, kQmin, kQmax);
    }
    gDw.weights = gW_dw;
    gDw.biases  = gB_dw;
    gDw.input_zero_point = static_cast<Q>(kZeroPoint);
    gDw.requantizers = gReqDw;

    // QPointwise (8 -> 16).
    for (std::size_t i = 0; i < QPwType::NumFilters; ++i) gB_pw[i] = 0;
    gPw.weights = gW_pw;
    gPw.biases  = gB_pw;
    gPw.input_zero_point = static_cast<Q>(kZeroPoint);
    gPw.requantizer = tinymind::buildRequantizer<Q>(
        dwScale, kWScale, pwScale, kZeroPoint, kQmin, kQmax);

    // QGlobalAvgPool: same scale on input and output, just clamp range.
    gGap.qmin = static_cast<Q>(kQmin);
    gGap.qmax = static_cast<Q>(kQmax);

    // QDense (16 -> 10) implemented as 1x1 pointwise; its input is the
    // average pool's output, at the pointwise layer's scale.
    for (std::size_t i = 0; i < QDenseType::NumFilters; ++i) gB_dense[i] = 0;
    gDense.weights = gW_dense;
    gDense.biases  = gB_dense;
    gDense.input_zero_point = static_cast<Q>(kZeroPoint);
    gDense.requantizer = tinymind::buildRequantizer<Q>(
        pwScale, kWScale, denseScale, kZeroPoint, kQmin, kQmax);

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

    // The int8 logits against the float model on the same input. The
    // weights are untrained, so the class index itself means nothing; what
    // the comparison shows is that the int8 pipeline tracks the float one.
    static FloatActivations ref;
    floatForward(gInputF, ref);

    float logits[kNumClasses];
    float maxErr = 0.0f;
    float refMin = ref.dense[0];
    float refMax = ref.dense[0];
    std::cout << "\nLogits (int8 -> dequantized, float reference):\n"
              << std::fixed << std::setprecision(6);
    for (std::size_t c = 0; c < kNumClasses; ++c)
    {
        logits[c] = tinymind::dequantize<Q>(gBufDense[c], denseScale, kZeroPoint);
        const float err = std::fabs(logits[c] - ref.dense[c]);
        if (err > maxErr) maxErr = err;
        if (ref.dense[c] < refMin) refMin = ref.dense[c];
        if (ref.dense[c] > refMax) refMax = ref.dense[c];
        std::cout << "  class " << c << ": " << std::setw(4) << static_cast<int>(gBufDense[c])
                  << " -> " << std::setw(9) << logits[c]
                  << "   float " << std::setw(9) << ref.dense[c] << "\n";
    }
    std::cout << "max |int8 - float| = " << maxErr << " ("
              << std::setprecision(1) << 100.0f * maxErr / (refMax - refMin)
              << "% of the float logits' range)\n"
              << "argmax (untrained weights): int8 " << argmaxOf(logits, kNumClasses)
              << ", float " << argmaxOf(ref.dense, kNumClasses) << "\n";

    return 0;
}
