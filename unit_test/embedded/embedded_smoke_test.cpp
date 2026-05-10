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

// Embedded smoke test.
//
// Drives the (FLOAT, STD, QUANT) build matrix configured by
// unit_test/embedded/Makefile:
//
//   FLOAT=0 STD=0            freestanding       - pure QValue pipeline, no
//                                                  hosted deps.
//   FLOAT=1 STD=0            no-stdlib FPU      - QValue + float forward-pass
//                                                  layers (Conv/Pool/Dropout
//                                                  inference). Verifies float
//                                                  ValueType doesn't drag in
//                                                  <cmath>.
//   FLOAT=0 STD=1            no-FPU hosted      - QValue pipeline, stdlib
//                                                  available.
//   FLOAT=1 STD=1            hosted             - everything.
//   FLOAT=0 STD=0 QUANT=1    quant freestanding - inference-only int8
//                                                  pipeline (parallel Q*
//                                                  layers from cpp/q*.hpp).
//                                                  Catches accidental float
//                                                  or stdlib leaks via the
//                                                  quantization headers.
//
// The QValue pipeline runs at every setting; the float pipeline only
// compiles in when FLOAT=1; the quantized pipeline only compiles in when
// QUANT=1. Calibration helpers (qcalibration.hpp) are intentionally NOT
// included here because they live behind FLOAT && STD; the inference
// path must be exercisable with neither.
//
// The point is regression-guarding: fail loudly if a future change
// reintroduces a hosted-only or FPU-only dependency into a header that
// embedded users rely on.

#include "tinymind_platform.hpp"

#include "qformat.hpp"
#include "conv1d.hpp"
#include "pool1d.hpp"
#include "conv2d.hpp"
#include "depthwiseconv2d.hpp"
#include "pointwiseconv2d.hpp"
#include "pool2d.hpp"
#include "batchnorm.hpp"
#include "dropout.hpp"
#include "binarylayer.hpp"
#include "ternarylayer.hpp"
#include "selfattention1d.hpp"
#include "fft1d.hpp"
#include "adam.hpp"
#include "rmsprop.hpp"
#include "nnproperties.hpp"
#include "networkStats.hpp"

#if TINYMIND_ENABLE_QUANTIZATION
#include "qaffine.hpp"
#include "qconv2d.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
#include "qpool2d.hpp"
#include "qdense.hpp"
#include "qactivations.hpp"
#endif

#include <cstddef>
#include <cstdint>

namespace {

typedef tinymind::QValue<8, 8, true> Q88;

// 1D pipeline.
typedef tinymind::Conv1D<Q88, 16, 3, 1, 4> Conv1Type;       // 16 -> 14, 4 filters
typedef tinymind::MaxPool1D<Q88, Conv1Type::OutputLength, 2, 2, 4> Pool1Type;

// 2D pipeline.
typedef tinymind::Conv2D<Q88, 8, 8, 1, 3, 3, 1, 1, 2> Conv2Type;
typedef tinymind::MaxPool2D<Q88, Conv2Type::OutputHeight, Conv2Type::OutputWidth, 2, 2, 2> Pool2Type;
typedef tinymind::DepthwiseConv2D<Q88, Pool2Type::OutputHeight, Pool2Type::OutputWidth, 2, 3, 3, 1, 1> DwType;
typedef tinymind::PointwiseConv2D<Q88, DwType::OutputHeight, DwType::OutputWidth, 2, 4> PwType;
typedef tinymind::GlobalAvgPool2D<Q88, PwType::OutputHeight, PwType::OutputWidth, 4> GapType;

// Small auxiliary layers.
typedef tinymind::BatchNorm1D<Q88, 4> BnType;
typedef tinymind::Dropout<Q88, 4, 25> DropType;
typedef tinymind::SelfAttention1D<Q88, 4, 4, 4> AttType;

Q88 input1d[16];
Q88 conv1Out[Conv1Type::OutputSize];
Q88 pool1Out[Pool1Type::OutputSize];

Q88 input2d[8 * 8];
Q88 conv2Out[Conv2Type::OutputSize];
Q88 pool2Out[Pool2Type::OutputSize];
Q88 dwOut[DwType::OutputSize];
Q88 pwOut[PwType::OutputSize];
Q88 gapOut[GapType::OutputSize];

Q88 bnIn[4];
Q88 bnOut[4];
Q88 dropOut[4];
Q88 attOut[AttType::OutputSize];

Conv1Type gConv1;
Pool1Type gPool1;
Conv2Type gConv2;
Pool2Type gPool2;
DwType    gDw;
PwType    gPw;
GapType   gGap;
BnType    gBn;
DropType  gDrop;
AttType   gAtt;

void clearTo(Q88* p, size_t n, int v)
{
    for (size_t i = 0; i < n; ++i)
    {
        p[i] = Q88(v, 0);
    }
}

// Exercise the QValue path of nnproperties' generic ValueParser /
// ValueConverter. These are the only specializations available with
// TINYMIND_ENABLE_FLOAT=0 and must remain compilable on embedded.
Q88 exerciseValueConverter()
{
    typedef tinymind::ValueConverter<Q88, Q88> ConverterType;
    return ConverterType::convertToDestinationType(Q88(1, 0));
}

#if TINYMIND_ENABLE_QUANTIZATION
// Quantized inference pipeline. Must compile with FLOAT=0, STD=0 (no
// calibration helpers; the deployable inference target carries only
// pre-calibrated multiplier/shift/zero_point triples). Layer dimensions
// are tiny so the static buffers stay small.
typedef int8_t  Q;
typedef int32_t A;

typedef tinymind::QConv2D<Q, Q, A, Q, 8, 8, 1, 3, 3, 1, 1, 2>           QConvType;
typedef tinymind::QMaxPool2D<Q, QConvType::OutputHeight,
                              QConvType::OutputWidth, 2, 2, 2>           QMaxType;
typedef tinymind::QDepthwiseConv2D<Q, Q, A, Q, QMaxType::OutputHeight,
                                    QMaxType::OutputWidth, 2, 3, 3, 1, 1> QDwType;
typedef tinymind::QPointwiseConv2D<Q, Q, A, Q, QDwType::OutputHeight,
                                    QDwType::OutputWidth, 2, 4>           QPwType;
typedef tinymind::QGlobalAvgPool2D<Q, A, QPwType::OutputHeight,
                                    QPwType::OutputWidth, 4>              QGapType;
typedef tinymind::QDense<Q, Q, A, Q, QGapType::OutputSize, 4>             QDenseType;

Q qInput[QConvType::InputSize];
Q qConvOut[QConvType::OutputSize];
Q qMaxOut[QMaxType::OutputSize];
Q qDwOut[QDwType::OutputSize];
Q qPwOut[QPwType::OutputSize];
Q qGapOut[QGapType::OutputSize];
Q qDenseOut[QDenseType::OutputLength];

Q qConvW[QConvType::TotalWeights];
A qConvB[QConvType::NumFilters];
Q qDwW[QDwType::TotalWeights];
A qDwB[QDwType::Channels];
tinymind::Requantizer<A, Q> qDwReq[QDwType::Channels];
Q qPwW[QPwType::TotalWeights];
A qPwB[QPwType::NumFilters];
Q qDenseW[QDenseType::InputLength * QDenseType::OutputLength];
A qDenseB[QDenseType::OutputLength];

QConvType  gQConv;
QMaxType   gQMax;
QDwType    gQDw;
QPwType    gQPw;
QGapType   gQGap;
QDenseType gQDense;

void clearI8(Q* p, size_t n)
{
    for (size_t i = 0; i < n; ++i) p[i] = 0;
}
void clearI32(A* p, size_t n)
{
    for (size_t i = 0; i < n; ++i) p[i] = 0;
}

bool exerciseQuantPipeline()
{
    tinymind::Requantizer<A, Q> r;
    r.multiplier = static_cast<int32_t>(1) << 30;
    r.shift = 0;
    r.zero_point = 0;
    r.qmin = -127;
    r.qmax = 127;

    clearI8(qInput, QConvType::InputSize);
    clearI8(qConvW, QConvType::TotalWeights);
    clearI32(qConvB, QConvType::NumFilters);
    clearI8(qDwW, QDwType::TotalWeights);
    clearI32(qDwB, QDwType::Channels);
    for (size_t i = 0; i < QDwType::Channels; ++i) qDwReq[i] = r;
    clearI8(qPwW, QPwType::TotalWeights);
    clearI32(qPwB, QPwType::NumFilters);
    clearI8(qDenseW, QDenseType::InputLength * QDenseType::OutputLength);
    clearI32(qDenseB, QDenseType::OutputLength);

    gQConv.weights = qConvW;
    gQConv.biases  = qConvB;
    gQConv.input_zero_point = 0;
    gQConv.requantizer = r;

    gQDw.weights = qDwW;
    gQDw.biases  = qDwB;
    gQDw.input_zero_point = 0;
    gQDw.requantizers = qDwReq;

    gQPw.weights = qPwW;
    gQPw.biases  = qPwB;
    gQPw.input_zero_point = 0;
    gQPw.requantizer = r;

    gQGap.qmin = -127;
    gQGap.qmax = 127;

    gQDense.weights = qDenseW;
    gQDense.biases  = qDenseB;
    gQDense.input_zero_point = 0;
    gQDense.requantizer = r;

    gQConv.forward(qInput, qConvOut);
    gQMax.forward(qConvOut, qMaxOut);
    gQDw.forward(qMaxOut, qDwOut);
    gQPw.forward(qDwOut, qPwOut);
    tinymind::qreluBuffer<Q>(qPwOut, QPwType::OutputSize, static_cast<Q>(0));
    gQGap.forward(qPwOut, qGapOut);
    gQDense.forward(qGapOut, qDenseOut);

    // Saturating Q0.31 multiply identity check.
    const int32_t s = tinymind::saturatingRoundingDoublingHighMul(
        static_cast<int32_t>(1) << 30, static_cast<int32_t>(1) << 30);

    return (qDenseOut[0] == qDenseOut[0]) && (s != 0);
}
#endif // TINYMIND_ENABLE_QUANTIZATION

#if TINYMIND_ENABLE_FLOAT
// Float pipeline: layers that do not call SquareRootApproximation<float> (so
// they are safe at FLOAT=1, STD=0). This catches accidental <cmath> leaks via
// the float-as-ValueType code paths.
typedef tinymind::Conv1D<float, 16, 3, 1, 4>                                       Conv1FloatType;
typedef tinymind::MaxPool1D<float, Conv1FloatType::OutputLength, 2, 2, 4>          Pool1FloatType;
typedef tinymind::Conv2D<float, 8, 8, 1, 3, 3, 1, 1, 2>                            Conv2FloatType;
typedef tinymind::MaxPool2D<float, Conv2FloatType::OutputHeight,
                                   Conv2FloatType::OutputWidth, 2, 2, 2>           Pool2FloatType;
typedef tinymind::Dropout<float, 4, 25>                                            DropFloatType;

float fInput1d[16];
float fConv1Out[Conv1FloatType::OutputSize];
float fPool1Out[Pool1FloatType::OutputSize];
float fInput2d[8 * 8];
float fConv2Out[Conv2FloatType::OutputSize];
float fPool2Out[Pool2FloatType::OutputSize];
float fDropIn[4];
float fDropOut[4];

Conv1FloatType gConv1f;
Pool1FloatType gPool1f;
Conv2FloatType gConv2f;
Pool2FloatType gPool2f;
DropFloatType  gDropf;

void clearFloat(float* p, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        p[i] = 0.0f;
    }
}
#endif // TINYMIND_ENABLE_FLOAT

} // namespace

int main()
{
    clearTo(input1d, 16, 0);
    clearTo(input2d, 8 * 8, 0);

    gConv1.forward(input1d, conv1Out);
    gPool1.forward(conv1Out, pool1Out);

    gConv2.forward(input2d, conv2Out);
    gPool2.forward(conv2Out, pool2Out);
    gDw.forward(pool2Out, dwOut);
    gPw.forward(dwOut, pwOut);
    gGap.forward(pwOut, gapOut);

    clearTo(bnIn, 4, 0);
    gBn.forward(bnIn, bnOut);

    gDrop.setTraining(false);
    gDrop.forward(bnOut, dropOut);

    Q88 attIn[AttType::InputSize];
    clearTo(attIn, AttType::InputSize, 0);
    gAtt.forward(attIn, attOut);

    const Q88 v = exerciseValueConverter();

    // ValueConverter<Q88,Q88> on Q88(1,0) must round-trip to a non-zero raw value.
    // gapOut[0] must equal itself (rules out ValueType being something exotic).
    bool ok = (v.getValue() != 0) && (gapOut[0] == gapOut[0]);

#if TINYMIND_ENABLE_QUANTIZATION
    ok = ok && exerciseQuantPipeline();
#endif

#if TINYMIND_ENABLE_FLOAT
    clearFloat(fInput1d, 16);
    clearFloat(fInput2d, 8 * 8);
    clearFloat(fDropIn, 4);

    gConv1f.forward(fInput1d, fConv1Out);
    gPool1f.forward(fConv1Out, fPool1Out);
    gConv2f.forward(fInput2d, fConv2Out);
    gPool2f.forward(fConv2Out, fPool2Out);
    gDropf.setTraining(false);
    gDropf.forward(fDropIn, fDropOut);

    ok = ok && (fPool1Out[0] == fPool1Out[0]) && (fPool2Out[0] == fPool2Out[0])
            && (fDropOut[0] == fDropOut[0]);
#endif

    return ok ? 0 : 1;
}
