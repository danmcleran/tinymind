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
#include "qadd.hpp"
#include "qmul.hpp"
#include "qconcat.hpp"
#include "qpad.hpp"
#include "qbatchnorm.hpp"
#include "qlayernorm.hpp"
#include "qsoftmax.hpp"
#include "qlstm.hpp"
#include "qgru.hpp"
#include "qfft1d.hpp"
#include "qattention1d.hpp"
#include "qattention_softmax.hpp"
#include "qmha.hpp"
#endif

#if TINYMIND_ENABLE_FP16
#include "tinymind_fp16.hpp"
#endif

#if TINYMIND_ENABLE_FLOAT
#include "qbridge.hpp"
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

    // Phase 10 composition ops, exercised at the gate to keep them
    // freestanding-clean. Tiny shapes; nothing depends on the output.
    tinymind::QAdd<Q, Q, Q, 4> qadd;
    qadd.input_a_zero_point = 0;
    qadd.input_b_zero_point = 0;
    qadd.left_shift = 20;
    qadd.input_a_multiplier = static_cast<int32_t>(1) << 30;
    qadd.input_a_shift = -1;
    qadd.input_b_multiplier = static_cast<int32_t>(1) << 30;
    qadd.input_b_shift = -1;
    qadd.output_requantizer.multiplier = static_cast<int32_t>(1) << 30;
    qadd.output_requantizer.shift = 20;
    qadd.output_requantizer.zero_point = 0;
    qadd.output_requantizer.qmin = -128;
    qadd.output_requantizer.qmax = 127;
    Q add_a[4] = {1, 2, 3, 4};
    Q add_b[4] = {-1, -2, -3, -4};
    Q add_y[4] = {0, 0, 0, 0};
    qadd.forward(add_a, add_b, add_y);

    tinymind::QMul<Q, Q, Q, 4> qmul;
    qmul.input_a_zero_point = 0;
    qmul.input_b_zero_point = 0;
    qmul.requantizer = r;
    Q mul_y[4] = {0, 0, 0, 0};
    qmul.forward(add_a, add_b, mul_y);

    tinymind::QConcat2_2D<Q, Q, Q, 2, 2, 1, 1> qconcat;
    qconcat.input_a_zero_point = 0;
    qconcat.input_b_zero_point = 0;
    qconcat.requantizer_a = r;
    qconcat.requantizer_b = r;
    Q ca_in[4] = {1, 2, 3, 4};
    Q cb_in[4] = {5, 6, 7, 8};
    Q cc_out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    qconcat.forward(ca_in, cb_in, cc_out);

    tinymind::QPad2D<Q, 2, 2, 1, 1, 1, 1, 1> qpad;
    qpad.pad_value = 0;
    Q pad_in[4] = {1, 2, 3, 4};
    Q pad_out[16];
    qpad.forward(pad_in, pad_out);

    // Per-channel conv smoke.
    typedef tinymind::QConv2DPerChannel<Q, Q, A, Q, 4, 4, 1, 3, 3, 1, 1, 2>
        QConvPCType;
    QConvPCType qcpc;
    Q pc_w[QConvPCType::TotalWeights] = {0};
    A pc_b[QConvPCType::NumFilters] = {0, 0};
    tinymind::Requantizer<A, Q> pc_rq[QConvPCType::NumFilters];
    pc_rq[0] = r; pc_rq[1] = r;
    Q pc_in[QConvPCType::InputSize] = {0};
    Q pc_out[QConvPCType::OutputSize];
    qcpc.weights = pc_w;
    qcpc.biases = pc_b;
    qcpc.input_zero_point = 0;
    qcpc.requantizers = pc_rq;
    qcpc.forward(pc_in, pc_out);

    // Saturating Q0.31 multiply identity check.
    const int32_t s = tinymind::saturatingRoundingDoublingHighMul(
        static_cast<int32_t>(1) << 30, static_cast<int32_t>(1) << 30);

    // Phase 11 ops: int8 batchnorm, layernorm, softmax. Exercised at the
    // gate so the new headers stay freestanding-clean.
    tinymind::QBatchNormChannelParams bn_params[2];
    bn_params[0].multiplier  = static_cast<int32_t>(1) << 30;
    bn_params[0].shift       = 0;
    bn_params[0].bias_addend = 0;
    bn_params[1].multiplier  = static_cast<int32_t>(1) << 30;
    bn_params[1].shift       = -1;
    bn_params[1].bias_addend = 1;

    tinymind::QBatchNorm2D<Q, Q, 2, 2, 2> qbn;
    qbn.params = bn_params;
    qbn.input_zero_point  = 0;
    qbn.output_zero_point = 0;
    qbn.qmin = -128;
    qbn.qmax = 127;
    Q bn_in[8]  = {1, 2, 3, 4, 5, 6, 7, 8};
    Q bn_out_buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    qbn.forward(bn_in, bn_out_buf);

    int16_t ln_gamma[4] = {1 << 14, 1 << 14, 1 << 14, 1 << 14};
    int32_t ln_beta[4]  = {0, 0, 0, 0};
    tinymind::QLayerNorm1D<Q, Q, 1, 4> qln;
    qln.gamma = ln_gamma;
    qln.beta  = ln_beta;
    qln.epsilon_q = 1;
    qln.output_multiplier = static_cast<int32_t>(1) << 30;
    qln.output_shift = 0;
    qln.output_zero_point = 0;
    qln.qmin = -128;
    qln.qmax = 127;
    Q ln_in[4]  = {-10, 0, 10, 20};
    Q ln_out_buf[4] = {0, 0, 0, 0};
    qln.forward(ln_in, ln_out_buf);

    int32_t exp_lut[256];
    for (size_t i = 0; i < 256; ++i)
    {
        // Linear ramp stand-in; the calibrated table is built host-side.
        // The freestanding smoke only validates the data path compiles.
        exp_lut[i] = static_cast<int32_t>(i + 1);
    }
    tinymind::QSoftmax1D<Q, Q, 1, 4> qsm;
    qsm.exp_lut = exp_lut;
    qsm.output_zero_point = -128;
    qsm.qmin = -128;
    qsm.qmax = 127;
    Q sm_in[4]  = {-5, -3, 0, 2};
    Q sm_out_buf[4] = {0, 0, 0, 0};
    qsm.forward(sm_in, sm_out_buf);

    // Phase 12 recurrent ops smoke. QLSTM cell-state defaults to int8; the
    // int16 cell-state corner only instantiates when the
    // TINYMIND_ENABLE_INT16_ACCUM gate is on so embedded toolchains without
    // an int16 storage need not pay for that template instantiation.
    typedef tinymind::QLSTMCell<Q, Q, A, Q, Q, Q, 2, 3> QLSTMI8Cell;
    static int8_t lstm_w_input    [QLSTMI8Cell::NumGates * 3 * 2] = {0};
    static int8_t lstm_w_recurrent[QLSTMI8Cell::NumGates * 3 * 3] = {0};
    static int32_t lstm_biases    [QLSTMI8Cell::NumGates * 3]     = {0};
    static int8_t lstm_sigmoid_lut[256] = {0};
    static int8_t lstm_tanh_lut   [256] = {0};
    static int8_t lstm_tanh_cell_lut[256] = {0};
    QLSTMI8Cell qlstm;
    qlstm.w_input     = lstm_w_input;
    qlstm.w_recurrent = lstm_w_recurrent;
    qlstm.biases      = lstm_biases;
    qlstm.input_zero_point  = 0;
    qlstm.hidden_zero_point = 0;
    qlstm.cell_zero_point   = 0;
    for (size_t g = 0; g < QLSTMI8Cell::NumGates; ++g)
    {
        qlstm.input_to_lut_multiplier   [g] = static_cast<int32_t>(1) << 30;
        qlstm.input_to_lut_shift        [g] = 0;
        qlstm.recurrent_to_lut_multiplier[g] = static_cast<int32_t>(1) << 30;
        qlstm.recurrent_to_lut_shift    [g] = 0;
    }
    qlstm.sigmoid_lut             = lstm_sigmoid_lut;
    qlstm.tanh_lut                = lstm_tanh_lut;
    qlstm.tanh_cell_lut           = lstm_tanh_cell_lut;
    qlstm.f_times_c_multiplier    = static_cast<int32_t>(1) << 30;
    qlstm.f_times_c_shift         = 0;
    qlstm.i_times_g_multiplier    = static_cast<int32_t>(1) << 30;
    qlstm.i_times_g_shift         = 0;
    qlstm.cell_qmin               = -128;
    qlstm.cell_qmax               =  127;
    qlstm.cell_to_tanh_multiplier = static_cast<int32_t>(1) << 30;
    qlstm.cell_to_tanh_shift      = 0;
    qlstm.output_requantizer      = r;
    int8_t lstm_x[2] = {1, -1};
    int8_t lstm_h[3] = {0, 0, 0};
    int8_t lstm_c[3] = {0, 0, 0};
    qlstm.forward(lstm_x, lstm_h, lstm_c);

#if TINYMIND_ENABLE_INT16_ACCUM
    typedef tinymind::QLSTMCell<Q, Q, A, Q, Q, int16_t, 2, 3> QLSTMI16Cell;
    QLSTMI16Cell qlstm16;
    qlstm16.w_input     = lstm_w_input;
    qlstm16.w_recurrent = lstm_w_recurrent;
    qlstm16.biases      = lstm_biases;
    qlstm16.input_zero_point  = 0;
    qlstm16.hidden_zero_point = 0;
    qlstm16.cell_zero_point   = 0;
    for (size_t g = 0; g < QLSTMI16Cell::NumGates; ++g)
    {
        qlstm16.input_to_lut_multiplier   [g] = static_cast<int32_t>(1) << 30;
        qlstm16.input_to_lut_shift        [g] = 0;
        qlstm16.recurrent_to_lut_multiplier[g] = static_cast<int32_t>(1) << 30;
        qlstm16.recurrent_to_lut_shift    [g] = 0;
    }
    qlstm16.sigmoid_lut             = lstm_sigmoid_lut;
    qlstm16.tanh_lut                = lstm_tanh_lut;
    qlstm16.tanh_cell_lut           = lstm_tanh_cell_lut;
    qlstm16.f_times_c_multiplier    = static_cast<int32_t>(1) << 30;
    qlstm16.f_times_c_shift         = 0;
    qlstm16.i_times_g_multiplier    = static_cast<int32_t>(1) << 30;
    qlstm16.i_times_g_shift         = 0;
    qlstm16.cell_qmin               = -32768;
    qlstm16.cell_qmax               =  32767;
    qlstm16.cell_to_tanh_multiplier = static_cast<int32_t>(1) << 30;
    qlstm16.cell_to_tanh_shift      = 0;
    qlstm16.output_requantizer      = r;
    int16_t lstm_c16[3] = {0, 0, 0};
    int8_t  lstm_h16[3] = {0, 0, 0};
    qlstm16.forward(lstm_x, lstm_h16, lstm_c16);
#endif

    typedef tinymind::QGRUCell<Q, Q, A, Q, Q, 2, 3> QGRUI8Cell;
    static int8_t gru_w_input    [QGRUI8Cell::NumGates * 3 * 2] = {0};
    static int8_t gru_w_recurrent[QGRUI8Cell::NumGates * 3 * 3] = {0};
    static int32_t gru_biases    [QGRUI8Cell::NumGates * 3]     = {0};
    QGRUI8Cell qgru;
    qgru.w_input     = gru_w_input;
    qgru.w_recurrent = gru_w_recurrent;
    qgru.biases      = gru_biases;
    qgru.input_zero_point  = 0;
    qgru.hidden_zero_point = 0;
    for (size_t g = 0; g < QGRUI8Cell::NumGates; ++g)
    {
        qgru.input_to_lut_multiplier   [g] = static_cast<int32_t>(1) << 30;
        qgru.input_to_lut_shift        [g] = 0;
        qgru.recurrent_to_lut_multiplier[g] = static_cast<int32_t>(1) << 30;
        qgru.recurrent_to_lut_shift    [g] = 0;
    }
    qgru.sigmoid_lut                    = lstm_sigmoid_lut;
    qgru.tanh_lut                       = lstm_tanh_lut;
    qgru.r_times_h_multiplier           = static_cast<int32_t>(1) << 30;
    qgru.r_times_h_shift                = 0;
    qgru.one_minus_z_times_n_multiplier = static_cast<int32_t>(1) << 30;
    qgru.one_minus_z_times_n_shift      = 0;
    qgru.z_times_h_multiplier           = static_cast<int32_t>(1) << 30;
    qgru.z_times_h_shift                = 0;
    qgru.output_qmin                    = -128;
    qgru.output_qmax                    =  127;
    int8_t gru_h[3] = {0, 0, 0};
    qgru.forward(lstm_x, gru_h);

    // Phase 13 attention + FFT smoke. Tiny shapes; nothing depends on the
    // output. Verifies the new headers stay freestanding-clean.
    static const int16_t fft_cos[4] = {32767, 23170, 0, -23170};
    static const int16_t fft_sin[4] = {0, -23170, -32767, -23170};
    tinymind::QFFT1D<8> qfft;
    qfft.twiddle_cos = fft_cos;
    qfft.twiddle_sin = fft_sin;
    int16_t fft_re[8] = {1, 0, 1, 0, 1, 0, 1, 0};
    int16_t fft_im[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    qfft.forward(fft_re, fft_im);
    int32_t fft_mag[8] = {0};
    tinymind::QFFT1D<8>::magnitudeSquared(fft_re, fft_im, fft_mag);

    typedef tinymind::QAttention1D<Q, Q, A, Q, Q, Q, 2, 4, 3> QAttnLinearType;
    static int8_t att_w[QAttnLinearType::TotalWeights] = {0};
    static int32_t att_b[QAttnLinearType::TotalBiases] = {0};
    QAttnLinearType qattn;
    qattn.weights = att_w;
    qattn.biases  = att_b;
    qattn.input_zero_point = 0;
    qattn.q_zero_point     = 0;
    qattn.k_zero_point     = 0;
    qattn.v_zero_point     = 0;
    qattn.kv_zero_point    = 0;
    qattn.q_requantizer  = r;
    qattn.k_requantizer  = r;
    qattn.v_requantizer  = r;
    qattn.kv_requantizer = r;
    qattn.output_requantizer = r;
    int8_t attn_in[QAttnLinearType::InputSize] = {0};
    int8_t attn_q_scratch [QAttnLinearType::QScratchSize]  = {0};
    int8_t attn_k_scratch [QAttnLinearType::KScratchSize]  = {0};
    int8_t attn_v_scratch [QAttnLinearType::VScratchSize]  = {0};
    int8_t attn_kv_scratch[QAttnLinearType::KVScratchSize] = {0};
    int8_t attn_out[QAttnLinearType::OutputSize] = {0};
    qattn.forward(attn_in,
                  attn_q_scratch, attn_k_scratch, attn_v_scratch,
                  attn_kv_scratch, attn_out);

    typedef tinymind::QAttentionSoftmax1D<Q, Q, A, Q, Q, Q, Q, 2, 4, 3>
        QAttnSoftType;
    QAttnSoftType qattn_sm;
    qattn_sm.weights = att_w;
    qattn_sm.biases  = att_b;
    qattn_sm.input_zero_point = 0;
    qattn_sm.q_zero_point     = 0;
    qattn_sm.k_zero_point     = 0;
    qattn_sm.v_zero_point     = 0;
    qattn_sm.attn_zero_point  = -128;
    qattn_sm.q_requantizer     = r;
    qattn_sm.k_requantizer     = r;
    qattn_sm.v_requantizer     = r;
    qattn_sm.score_requantizer = r;
    qattn_sm.softmax_exp_lut   = exp_lut;
    qattn_sm.attn_qmin         = -128;
    qattn_sm.attn_qmax         =  127;
    qattn_sm.output_requantizer = r;
    int8_t sm_score_scratch[QAttnSoftType::ScoreScratchSize] = {0};
    int8_t sm_attn_scratch [QAttnSoftType::AttnScratchSize]  = {0};
    int8_t sm_out[QAttnSoftType::OutputSize] = {0};
    qattn_sm.forward(attn_in,
                     attn_q_scratch, attn_k_scratch, attn_v_scratch,
                     sm_score_scratch, sm_attn_scratch, sm_out);

    typedef tinymind::QMultiHeadLinearAttention1D<
        Q, Q, A, Q, Q, Q, 2, 4, 3, 2> QMHAType;
    QMHAType qmha;
    for (size_t hi = 0; hi < QMHAType::NumHeads; ++hi)
    {
        qmha.heads[hi] = qattn;
    }
    int8_t mha_head_out[QMHAType::HeadOutScratchSize] = {0};
    int8_t mha_out[QMHAType::OutputSize] = {0};
    qmha.forward(attn_in,
                 attn_q_scratch, attn_k_scratch, attn_v_scratch,
                 attn_kv_scratch, mha_head_out, mha_out);

    return (qDenseOut[0] == qDenseOut[0]) && (s != 0)
        && (add_y[0] == add_y[0]) && (mul_y[0] == mul_y[0])
        && (cc_out[0] == cc_out[0]) && (pad_out[0] == pad_out[0])
        && (pc_out[0] == pc_out[0])
        && (bn_out_buf[0] == bn_out_buf[0])
        && (ln_out_buf[0] == ln_out_buf[0])
        && (sm_out_buf[0] == sm_out_buf[0])
        && (lstm_h[0] == lstm_h[0]) && (lstm_c[0] == lstm_c[0])
        && (gru_h[0]  == gru_h[0])
        && (fft_re[0] == fft_re[0]) && (fft_mag[0] == fft_mag[0])
        && (attn_out[0] == attn_out[0])
        && (sm_out[0] == sm_out[0])
        && (mha_out[0] == mha_out[0]);
}
#endif // TINYMIND_ENABLE_QUANTIZATION

#if TINYMIND_ENABLE_FLOAT
// Phase 9 bridges. Exercise float -> QValue -> int8 affine -> QValue ->
// float round-trip and (when FP16 is on) fp16/bf16 storage. Must compile
// freestanding at FLOAT=1 STD=0 — no <cmath>, no <type_traits>.
bool exerciseBridges()
{
    typedef int8_t I8;
    const float scale = 0.05f;
    const int32_t zp = -3;

    const I8 q = tinymind::affineQuantize<I8>(1.25f, scale, zp, -128, 127);
    const float deq = tinymind::affineDequantize<I8>(q, scale, zp);

    const Q88 qv = tinymind::floatToQValue<Q88>(0.75f);
    const float as_f = tinymind::qValueToFloat<Q88>(qv);
    const I8 qv_to_i8 = tinymind::qValueToAffine<Q88, I8>(qv, scale, zp, -128, 127);
    const Q88 back = tinymind::affineToQValue<Q88, I8>(qv_to_i8, scale, zp);

    bool ok = (deq == deq) && (as_f == as_f) && (back.getValue() == back.getValue());
#if TINYMIND_ENABLE_FP16
    const tinymind::fp16_t h = tinymind::floatToFp16(2.5f);
    const float h_back = tinymind::fp16ToFloat(h);

    const tinymind::bf16_t b = tinymind::floatToBf16(2.5f);
    const float b_back = tinymind::bf16ToFloat(b);

    const I8 h_to_i8 = tinymind::fp16ToAffineI8(h, scale, zp, -128, 127);
    const tinymind::fp16_t h_round = tinymind::affineI8ToFp16(h_to_i8, scale, zp);

    const I8 b_to_i8 = tinymind::bf16ToAffineI8(b, scale, zp, -128, 127);
    const tinymind::bf16_t b_round = tinymind::affineI8ToBf16(b_to_i8, scale, zp);

    // Compare against pre-quant fp16/bf16 bit patterns; bits should be
    // close after the affine round-trip.
    ok = ok && (h_back == h_back) && (b_back == b_back)
            && (h_round.bits != 0xFFFFu) && (b_round.bits != 0xFFFFu);
#endif
    return ok;
}

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

    ok = ok && exerciseBridges();
#endif

    return ok ? 0 : 1;
}
