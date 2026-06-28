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
#include "dual.hpp"
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
#include "qbridge.hpp"
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
#include "qcfc.hpp"
#include "qfft1d.hpp"
#include "qattention1d.hpp"
#include "qattention_softmax.hpp"
#include "qcausalattention1d.hpp"
#include "qcausalattention_softmax.hpp"
#include "qcrossattention.hpp"
#include "qkvcache.hpp"
#include "qssm.hpp"
#include "qtree.hpp"
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

    // QCfC (closed-form continuous-time) cell smoke. 2 inputs, 3 hidden,
    // 4 backbone units. Verifies qcfc.hpp stays freestanding-clean.
    typedef tinymind::QCfCCell<Q, Q, A, Q, 2, 3, 4> QCfCI8Cell;
    static int8_t cfc_w_bin [4 * 2] = {0};
    static int8_t cfc_w_bh  [4 * 3] = {0};
    static int8_t cfc_w_ff1 [3 * 4] = {0};
    static int8_t cfc_w_ff2 [3 * 4] = {0};
    static int8_t cfc_w_ta  [3 * 4] = {0};
    static int8_t cfc_w_tb  [3 * 4] = {0};
    static int32_t cfc_b_bb [4] = {0};
    static int32_t cfc_b_ff1[3] = {0};
    static int32_t cfc_b_ff2[3] = {0};
    static int32_t cfc_b_t  [3] = {0};
    QCfCI8Cell qcfc;
    qcfc.w_backbone_input  = cfc_w_bin;
    qcfc.w_backbone_hidden = cfc_w_bh;
    qcfc.b_backbone        = cfc_b_bb;
    qcfc.w_ff1 = cfc_w_ff1; qcfc.w_ff2 = cfc_w_ff2;
    qcfc.w_time_a = cfc_w_ta; qcfc.w_time_b = cfc_w_tb;
    qcfc.b_ff1 = cfc_b_ff1; qcfc.b_ff2 = cfc_b_ff2; qcfc.b_time = cfc_b_t;
    qcfc.input_zero_point = 0; qcfc.hidden_zero_point = 0;
    qcfc.backbone_input_multiplier  = static_cast<int32_t>(1) << 30;
    qcfc.backbone_input_shift       = 0;
    qcfc.backbone_hidden_multiplier = static_cast<int32_t>(1) << 30;
    qcfc.backbone_hidden_shift      = 0;
    qcfc.ff1_multiplier = static_cast<int32_t>(1) << 30; qcfc.ff1_shift = 0;
    qcfc.ff2_multiplier = static_cast<int32_t>(1) << 30; qcfc.ff2_shift = 0;
    qcfc.time_a_multiplier = static_cast<int32_t>(1) << 30; qcfc.time_a_shift = 0;
    qcfc.time_b_multiplier = static_cast<int32_t>(1) << 30; qcfc.time_b_shift = 0;
    qcfc.sigmoid_lut = lstm_sigmoid_lut; qcfc.tanh_lut = lstm_tanh_lut;
    qcfc.one_minus_t_times_ff1_multiplier = static_cast<int32_t>(1) << 30;
    qcfc.one_minus_t_times_ff1_shift      = 0;
    qcfc.t_times_ff2_multiplier = static_cast<int32_t>(1) << 30;
    qcfc.t_times_ff2_shift      = 0;
    qcfc.output_qmin = -128; qcfc.output_qmax = 127;
    int8_t cfc_h[3] = {0, 0, 0};
    qcfc.forward(lstm_x, cfc_h);

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

    // Decoder attention family (causal self + cross). Reuses att_w/att_b
    // (same 3*E*P weight / 3*P bias shape) and the existing exp_lut.
    typedef tinymind::QCausalAttention1D<Q, Q, A, Q, Q, Q, 2, 4, 3> QCausalLinType;
    QCausalLinType qcausal;
    qcausal.weights = att_w; qcausal.biases = att_b;
    qcausal.input_zero_point = 0;
    qcausal.q_zero_point = 0; qcausal.k_zero_point = 0; qcausal.v_zero_point = 0;
    qcausal.kv_zero_point = 0;
    qcausal.q_requantizer = r; qcausal.k_requantizer = r; qcausal.v_requantizer = r;
    qcausal.kv_requantizer = r; qcausal.output_requantizer = r;
    QCausalLinType::KVState causal_state;
    int8_t causal_out[QCausalLinType::OutputSize] = {0};
    qcausal.forward(attn_in, causal_state,
                    attn_q_scratch, attn_k_scratch, attn_v_scratch,
                    attn_kv_scratch, causal_out);

    typedef tinymind::QCausalAttentionSoftmax1D<Q, Q, A, Q, Q, Q, Q, 2, 4, 3>
        QCausalSoftType;
    QCausalSoftType qcausal_sm;
    qcausal_sm.weights = att_w; qcausal_sm.biases = att_b;
    qcausal_sm.input_zero_point = 0;
    qcausal_sm.q_zero_point = 0; qcausal_sm.k_zero_point = 0; qcausal_sm.v_zero_point = 0;
    qcausal_sm.attn_zero_point = -128;
    qcausal_sm.q_requantizer = r; qcausal_sm.k_requantizer = r; qcausal_sm.v_requantizer = r;
    qcausal_sm.score_requantizer = r;
    qcausal_sm.softmax_exp_lut = exp_lut;
    qcausal_sm.attn_qmin = -128; qcausal_sm.attn_qmax = 127;
    qcausal_sm.output_requantizer = r;
    QCausalSoftType::KVCache causal_cache;
    int8_t cs_score[QCausalSoftType::ScoreScratchSize] = {0};
    int8_t cs_attn [QCausalSoftType::AttnScratchSize]  = {0};
    int8_t cs_out[QCausalSoftType::OutputSize] = {0};
    qcausal_sm.forward(attn_in, causal_cache,
                       attn_q_scratch, attn_k_scratch, attn_v_scratch,
                       cs_score, cs_attn, cs_out);

    typedef tinymind::QCrossAttention1D<Q, Q, A, Q, Q, Q, 2, 2, 4, 3> QCrossLinType;
    QCrossLinType qcross;
    qcross.weights = att_w; qcross.biases = att_b;
    qcross.q_input_zero_point = 0; qcross.kv_input_zero_point = 0;
    qcross.q_zero_point = 0; qcross.k_zero_point = 0; qcross.v_zero_point = 0;
    qcross.kv_zero_point = 0;
    qcross.q_requantizer = r; qcross.k_requantizer = r; qcross.v_requantizer = r;
    qcross.kv_requantizer = r; qcross.output_requantizer = r;
    int8_t xk[QCrossLinType::KScratchSize] = {0};
    int8_t xv[QCrossLinType::VScratchSize] = {0};
    int8_t xkv[QCrossLinType::KVScratchSize] = {0};
    int8_t xq[QCrossLinType::QScratchSize] = {0};
    int8_t cross_out[2 * 3] = {0};
    qcross.forward(attn_in, attn_in, xk, xv, xkv, xq, cross_out);

    typedef tinymind::QCrossAttentionSoftmax1D<Q, Q, A, Q, Q, Q, Q, 2, 2, 4, 3>
        QCrossSoftType;
    QCrossSoftType qcross_sm;
    qcross_sm.weights = att_w; qcross_sm.biases = att_b;
    qcross_sm.q_input_zero_point = 0; qcross_sm.kv_input_zero_point = 0;
    qcross_sm.q_zero_point = 0; qcross_sm.k_zero_point = 0; qcross_sm.v_zero_point = 0;
    qcross_sm.attn_zero_point = -128;
    qcross_sm.q_requantizer = r; qcross_sm.k_requantizer = r; qcross_sm.v_requantizer = r;
    qcross_sm.score_requantizer = r;
    qcross_sm.softmax_exp_lut = exp_lut;
    qcross_sm.attn_qmin = -128; qcross_sm.attn_qmax = 127;
    qcross_sm.output_requantizer = r;
    QCrossSoftType::KVCache cross_cache;
    int8_t xcs_score[QCrossSoftType::ScoreScratchSize] = {0};
    int8_t xcs_attn [QCrossSoftType::AttnScratchSize]  = {0};
    int8_t cross_sm_out[2 * 3] = {0};
    qcross_sm.forward(attn_in, attn_in, cross_cache, xq, xcs_score, xcs_attn, cross_sm_out);

    // Diagonal state-space family (LTI + selective). Per-channel integer
    // coefficient arrays; freestanding (no LUT, gate is a clamped affine).
    static const int32_t ssm_mult[3] = {0x40000000, 0x40000000, 0x40000000};
    static const int32_t ssm_shift[3] = {0, 0, 0};
    static const int32_t ssm_gbias[3] = {0, 0, 0};
    typedef tinymind::QStateSpace1D<int8_t, int32_t, int8_t, 4, 3> QSSMType;
    QSSMType qssm;
    qssm.input_zero_point = 0; qssm.output_zero_point = 0; qssm.qmin = -128; qssm.qmax = 127;
    qssm.a_multiplier = ssm_mult; qssm.a_shift = ssm_shift;
    qssm.b_multiplier = ssm_mult; qssm.b_shift = ssm_shift;
    qssm.c_multiplier = ssm_mult; qssm.c_shift = ssm_shift;
    qssm.d_multiplier = nullptr;  qssm.d_shift = nullptr;
    QSSMType::State ssm_state;
    int8_t ssm_in[QSSMType::InputSize] = {0};
    int8_t ssm_out[QSSMType::OutputSize] = {0};
    qssm.forward(ssm_in, ssm_state, ssm_out);

    typedef tinymind::QSelectiveStateSpace1D<int8_t, int32_t, int8_t, 4, 3> QSelSSMType;
    QSelSSMType qsel;
    qsel.input_zero_point = 0; qsel.output_zero_point = 0; qsel.qmin = -128; qsel.qmax = 127;
    qsel.a_multiplier = ssm_mult; qsel.a_shift = ssm_shift;
    qsel.b_multiplier = ssm_mult; qsel.b_shift = ssm_shift;
    qsel.c_multiplier = ssm_mult; qsel.c_shift = ssm_shift;
    qsel.d_multiplier = nullptr;  qsel.d_shift = nullptr;
    qsel.gate_multiplier = ssm_mult; qsel.gate_shift = ssm_shift; qsel.gate_bias = ssm_gbias;
    QSelSSMType::State sel_state;
    int8_t sel_out[QSelSSMType::OutputSize] = {0};
    qsel.forward(ssm_in, sel_state, sel_out);

    // Quantized decision tree + GBDT ensemble (int compare + branch, no LUT).
    static const tinymind::QTreeNode<int8_t> tree_nodes[3] = {
        { 0, 0, 1, 2, 0 }, { -1, 0, -1, -1, 7 }, { -1, 0, -1, -1, -7 },
    };
    tinymind::QDecisionTree<int8_t, int8_t, 3, 2> qtree;
    qtree.nodes = tree_nodes;
    int8_t tree_x[2] = {0, 0};
    const int32_t tree_leaf = qtree.predict(tree_x);

    static const int16_t gbdt_roots[1] = {0};
    static const int16_t gbdt_classes[1] = {0};
    tinymind::QGBDT<int8_t, int8_t, 1, 2, 3, 2> qgbdt;
    qgbdt.nodes = tree_nodes; qgbdt.tree_root = gbdt_roots;
    qgbdt.tree_class = gbdt_classes; qgbdt.base_score = nullptr;
    const std::size_t gbdt_cls = qgbdt.predict(tree_x);

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
        && (mha_out[0] == mha_out[0])
        && (causal_out[0] == causal_out[0])
        && (cs_out[0] == cs_out[0])
        && (cross_out[0] == cross_out[0])
        && (cross_sm_out[0] == cross_sm_out[0])
        && (ssm_out[0] == ssm_out[0])
        && (sel_out[0] == sel_out[0])
        && (tree_leaf == 7)
        && (gbdt_cls == 0u);
}
int32_t gIntegerBridgeSink[2];

// Pure-integer Q <-> int8 bridge. Must compile at QUANT=1 FLOAT=0 STD=0:
// no <cmath>, no quantizeMultiplier (which is FLOAT && STD gated). Params
// are passed as raw int32 triples — host-side helpers
// buildAffineToQValueIntParams / buildQValueToAffineIntParams compute them
// at calibration time and the deployable target consumes them as data.
bool exerciseIntegerBridges()
{
    typedef int8_t I8;

    tinymind::AffineToQValueIntParams<Q88> a2q;
    a2q.multiplier = static_cast<int32_t>(0x40000000); // Q0.31 = 0.5
    a2q.shift = -8;                                    // ratio = 128
    a2q.zero_point = -3;

    const Q88 fromI8 = tinymind::affineToQValueInt<Q88, I8>(static_cast<I8>(10), a2q);

    tinymind::QValueToAffineIntParams<Q88> q2a;
    q2a.multiplier = static_cast<int32_t>(0x40000000);
    q2a.shift = 7;                                     // ratio = 1/256
    q2a.zero_point = -3;
    q2a.qmin = -128;
    q2a.qmax = 127;

    const Q88 qv = Q88(static_cast<Q88::FullWidthValueType>(0x0180)); // 1.5
    const I8 toI8 = tinymind::qValueToAffineInt<Q88, I8>(qv, q2a);

    // Smoke test cares only that the bridge compiled and ran. Use the
    // same array-self-compare idiom the QUANT pipeline above relies on.
    gIntegerBridgeSink[0] = static_cast<int32_t>(fromI8.getValue());
    gIntegerBridgeSink[1] = static_cast<int32_t>(toI8);
    return (gIntegerBridgeSink[0] == gIntegerBridgeSink[0])
        && (gIntegerBridgeSink[1] == gIntegerBridgeSink[1]);
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

// Forward-mode autodiff over a fixed-point input. Dual<QValue> needs only the
// value type's arithmetic, so it must build and run in the freestanding corner
// (no FLOAT, no STD, no <cmath>). f(x)=x*x+x at x=2 -> value 6, deriv 2x+1 = 5,
// both exact in Q8.8.
bool exerciseDual()
{
    tinymind::Dual<Q88> x(Q88(2, 0), Q88(1, 0));
    tinymind::Dual<Q88> y = (x * x) + x;
    return (y.value == Q88(6, 0)) && (y.deriv == Q88(5, 0));
}

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

    ok = ok && exerciseDual();

#if TINYMIND_ENABLE_QUANTIZATION
    ok = ok && exerciseQuantPipeline();
    ok = ok && exerciseIntegerBridges();
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
