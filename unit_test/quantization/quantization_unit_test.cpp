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

// Phase 1 quantization unit tests.
//
// Exercises the integer requantization primitives in cpp/qaffine.hpp:
//
//   * SaturatingRoundingDoublingHighMul edge cases (INT32_MIN x INT32_MIN
//     saturates, sign of nudge follows sign of product).
//   * RoundingDivideByPOT (no-op for shift<=0, rounding semantics for >0).
//   * quantizeMultiplier round-trips for ratios <1, ==1, >1, and 0.0.
//   * Requantizer.apply for identity, halving, doubling, zero-point bias,
//     and saturation at the destination storage bounds.
//   * QAffineTensor float scale field present under
//     TINYMIND_ENABLE_FLOAT=1.

#include <cstdint>
#include <climits>
#include <cmath>

#include "qaffine.hpp"
#include "qcalibration.hpp"
#include "qactivations.hpp"
#include "qdense.hpp"
#include "qmoe.hpp"
#include "qconv2d.hpp"
#include "qpool2d.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
#include "qbridge.hpp"
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
#include "qmha.hpp"
#include "qembedding.hpp"
#include "qpositional.hpp"
#include "qformat.hpp"
#include "include/tinymind_fp16.hpp"
#include "include/simd/simd_dispatch.hpp"
#include "compiler.h"

#define BOOST_TEST_MODULE quantization_unit_test
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING("-Wdangling-reference")
#include <boost/test/included/unit_test.hpp>
TINYMIND_DISABLE_WARNING_POP

using tinymind::saturatingRoundingDoublingHighMul;
using tinymind::roundingDivideByPOT;
using tinymind::Requantizer;
using tinymind::QAffineTensor;
using tinymind::qrelu;
using tinymind::qreluBuffer;
using tinymind::qrelu6;
using tinymind::qrelu6Buffer;
using tinymind::clampForRelu;
using tinymind::clampForRelu6;
using tinymind::QDense;
using tinymind::QMixtureOfExperts;
using tinymind::QTopKMixtureOfExperts;
using tinymind::QConv2D;
using tinymind::QMaxPool2D;
using tinymind::QAvgPool2D;
using tinymind::QGlobalAvgPool2D;
using tinymind::QDepthwiseConv2D;
using tinymind::QPointwiseConv2D;
using tinymind::QConv2DPerChannel;
using tinymind::QPointwiseConv2DPerChannel;
using tinymind::QAdd;
using tinymind::QMul;
using tinymind::QConcat2_2D;
using tinymind::QPad2D;
using tinymind::QBatchNorm1D;
using tinymind::QBatchNorm2D;
using tinymind::QBatchNormChannelParams;
using tinymind::QLayerNorm1D;
using tinymind::QSoftmax1D;
using tinymind::qIntegerSqrt64;
using tinymind::qInvSqrtQ30;
#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
using tinymind::quantizeMultiplier;
using tinymind::AffineParams;
using tinymind::RangeObserver;
using tinymind::computeAffineParamsAsymmetric;
using tinymind::computeAffineParamsSymmetric;
using tinymind::quantize;
using tinymind::dequantize;
using tinymind::quantizeBuffer;
using tinymind::dequantizeBuffer;
using tinymind::computePerChannelSymmetricScales;
using tinymind::buildRequantizer;
using tinymind::computeQuantizedSix;
using tinymind::computeQuantizedThreshold;
using tinymind::buildQSigmoidLUT;
using tinymind::buildQTanhLUT;
using tinymind::buildRescaler;
using tinymind::buildQAddParams;
using tinymind::QAddParams;
using tinymind::buildQMulRequantizer;
using tinymind::foldBatchNorm;
using tinymind::buildQBatchNormChannelParams;
using tinymind::buildQSoftmaxExpLUT;
using tinymind::quantizeLayerNormGamma;
using tinymind::quantizeLayerNormBeta;
using tinymind::buildQLayerNormOutputParams;
using tinymind::quantizeLayerNormEpsilon;
using tinymind::kQSoftmaxExpLUTSize;
using tinymind::QLSTMScales;
using tinymind::QLSTMParams;
using tinymind::buildQLSTMParams;
using tinymind::quantizeQLSTMBiases;
using tinymind::QGRUScales;
using tinymind::QGRUParams;
using tinymind::buildQGRUParams;
using tinymind::quantizeQGRUBiases;
using tinymind::QCfCScales;
using tinymind::QCfCParams;
using tinymind::buildQCfCParams;
using tinymind::quantizeQCfCBias;
using tinymind::quantizeQCfCTimeBias;
using tinymind::buildQFFTTwiddles;
using tinymind::qAttentionInvSqrt;
using tinymind::PercentileObserver;
using tinymind::KLDivergenceObserver;
using tinymind::crossLayerEqualizeDense;
using tinymind::crossLayerEqualizeConv2D;
#endif
using tinymind::kQActivationLUTSize;
using tinymind::qActivationLUTIndex;
using tinymind::qApplyLUT;
using tinymind::qApplyLUTBuffer;

namespace {

constexpr int32_t kQ31One = static_cast<int32_t>(0x7FFFFFFF); // ~ 1.0 in Q0.31

// Construct a Requantizer for symmetric int8 output (qmin=-128, qmax=127)
// from a Q0.31 multiplier and a shift, with zero_point = 0.
Requantizer<int32_t, int8_t> makeI8Requant(int32_t multiplier, int32_t shift, int8_t zp = 0)
{
    Requantizer<int32_t, int8_t> r;
    r.multiplier = multiplier;
    r.shift = shift;
    r.zero_point = zp;
    r.qmin = static_cast<int8_t>(-128);
    r.qmax = static_cast<int8_t>(127);
    return r;
}

} // namespace

BOOST_AUTO_TEST_SUITE(quantization)

BOOST_AUTO_TEST_CASE(saturating_high_mul_basic)
{
    // Q31 1.0 (kQ31One ~= 2^31 - 1) multiplied by any int32 returns
    // approximately the input (off by ~1 ULP because kQ31One is just
    // under 1.0 and rounding favors nearest).
    BOOST_TEST(saturatingRoundingDoublingHighMul(0, kQ31One) == 0);
    BOOST_TEST(saturatingRoundingDoublingHighMul(1000, kQ31One) == 1000);
    BOOST_TEST(saturatingRoundingDoublingHighMul(-1000, kQ31One) == -1000);

    // Q31 0.5 == 1<<30. Multiplying doubles and shifts: result is input/2.
    const int32_t half = static_cast<int32_t>(1) << 30;
    BOOST_TEST(saturatingRoundingDoublingHighMul(1000, half) == 500);
    BOOST_TEST(saturatingRoundingDoublingHighMul(-1000, half) == -500);
}

BOOST_AUTO_TEST_CASE(quantize_multiplier_renormalizes_rounded_up_mantissa)
{
    using tinymind::quantizeMultiplier;

    // A ratio whose frexp mantissa rounds up to exactly 2^31 trips the
    // renormalize edge case: the multiplier is halved and the shift bumped.
    int32_t mult = 0, shift = 0;
    quantizeMultiplier(1.0 - 1e-10, mult, shift);
    BOOST_TEST(mult == (static_cast<int32_t>(1) << 30));
    BOOST_TEST(shift == -1);

    // Sanity: the zero-ratio fast path is unaffected.
    quantizeMultiplier(0.0, mult, shift);
    BOOST_TEST(mult == 0);
    BOOST_TEST(shift == 0);
}

BOOST_AUTO_TEST_CASE(saturating_high_mul_int_min_squared_saturates)
{
    // The one overflow case in gemmlowp: INT32_MIN * INT32_MIN would yield
    // 2^62, doubled to 2^63 which doesn't fit in int32; saturate to MAX.
    const int32_t imin = static_cast<int32_t>(0x80000000);
    BOOST_TEST(saturatingRoundingDoublingHighMul(imin, imin) == static_cast<int32_t>(0x7FFFFFFF));
}

BOOST_AUTO_TEST_CASE(rounding_divide_by_pot_noop_for_zero_or_negative)
{
    BOOST_TEST(roundingDivideByPOT(1234, 0) == 1234);
    BOOST_TEST(roundingDivideByPOT(-1234, 0) == -1234);
    BOOST_TEST(roundingDivideByPOT(1234, -3) == 1234);
}

BOOST_AUTO_TEST_CASE(rounding_divide_by_pot_rounds_to_nearest)
{
    // Divide by 8 with rounding. Threshold is mask>>1 = 3, so remainders
    // > 3 round up.
    BOOST_TEST(roundingDivideByPOT(8, 3) == 1);    // exactly 1
    BOOST_TEST(roundingDivideByPOT(11, 3) == 1);   // 11/8=1 rem 3, no round
    BOOST_TEST(roundingDivideByPOT(12, 3) == 2);   // 12/8=1 rem 4, round up
    BOOST_TEST(roundingDivideByPOT(15, 3) == 2);   // 15/8=1 rem 7, round up
    BOOST_TEST(roundingDivideByPOT(-8, 3) == -1);
    // -12/8 = -1.5; gemmlowp rounds negative halves toward -inf -> -2.
    BOOST_TEST(roundingDivideByPOT(-12, 3) == -2);
    BOOST_TEST(roundingDivideByPOT(-11, 3) == -1); // -1.375 -> -1
    BOOST_TEST(roundingDivideByPOT(-13, 3) == -2); // -1.625 -> -2
}

BOOST_AUTO_TEST_CASE(requantizer_identity)
{
    // multiplier == kQ31One, shift == 0: rescale by ~1.0. Output equals
    // input clamped to int8 range.
    auto r = makeI8Requant(kQ31One, 0, 0);
    BOOST_TEST(static_cast<int>(r.apply(0)) == 0);
    BOOST_TEST(static_cast<int>(r.apply(50)) == 50);
    BOOST_TEST(static_cast<int>(r.apply(-50)) == -50);
}

BOOST_AUTO_TEST_CASE(requantizer_halve)
{
    // multiplier == kQ31One, shift == 1: divide-by-two with rounding.
    auto r = makeI8Requant(kQ31One, 1, 0);
    BOOST_TEST(static_cast<int>(r.apply(100)) == 50);
    BOOST_TEST(static_cast<int>(r.apply(-100)) == -50);
    BOOST_TEST(static_cast<int>(r.apply(0)) == 0);
}

BOOST_AUTO_TEST_CASE(requantizer_double_via_left_shift)
{
    // multiplier == kQ31One, shift == -1: left-shift accumulator before
    // multiply, doubling the effective scale.
    auto r = makeI8Requant(kQ31One, -1, 0);
    BOOST_TEST(static_cast<int>(r.apply(30)) == 60);
    BOOST_TEST(static_cast<int>(r.apply(-30)) == -60);
}

BOOST_AUTO_TEST_CASE(requantizer_saturates_at_qmin_qmax)
{
    auto r = makeI8Requant(kQ31One, 0, 0);
    BOOST_TEST(static_cast<int>(r.apply(1000)) == 127);    // saturate high
    BOOST_TEST(static_cast<int>(r.apply(-1000)) == -128);  // saturate low
}

BOOST_AUTO_TEST_CASE(requantizer_applies_zero_point)
{
    // zero_point=10 shifts every output up by 10 before saturation.
    auto r = makeI8Requant(kQ31One, 0, 10);
    BOOST_TEST(static_cast<int>(r.apply(0)) == 10);
    BOOST_TEST(static_cast<int>(r.apply(50)) == 60);
    BOOST_TEST(static_cast<int>(r.apply(-50)) == -40);

    // Saturation still respects qmin/qmax post-zero-point add.
    BOOST_TEST(static_cast<int>(r.apply(200)) == 127);
    BOOST_TEST(static_cast<int>(r.apply(-200)) == -128);
}

BOOST_AUTO_TEST_CASE(qaffine_tensor_holds_zero_point)
{
    QAffineTensor<int8_t> t{};
#if TINYMIND_ENABLE_FLOAT
    t.scale = 0.0625f;
    BOOST_TEST(t.scale == 0.0625f);
#endif
    t.zero_point = static_cast<int8_t>(-3);
    BOOST_TEST(static_cast<int>(t.zero_point) == -3);
}

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
BOOST_AUTO_TEST_CASE(quantize_multiplier_unity)
{
    int32_t m = 0;
    int32_t s = 0;
    quantizeMultiplier(1.0, m, s);
    // 1.0 -> mantissa 0.5, exponent 1; mantissa*2^31 == 2^30; renormalized
    // by the q == 2^31 branch isn't taken (q == 2^30 here). shift = -1
    // means left-shift by 1, then Q31-mul-by-0.5 yields the original.
    BOOST_TEST(m == (static_cast<int32_t>(1) << 30));
    BOOST_TEST(s == -1);

    Requantizer<int32_t, int8_t> r;
    r.multiplier = m;
    r.shift = s;
    r.zero_point = 0;
    r.qmin = -128;
    r.qmax = 127;
    BOOST_TEST(static_cast<int>(r.apply(42)) == 42);
}

BOOST_AUTO_TEST_CASE(quantize_multiplier_half)
{
    int32_t m = 0;
    int32_t s = 0;
    quantizeMultiplier(0.5, m, s);
    // 0.5 -> mantissa 0.5, exponent 0; q == 2^30, shift == 0.
    BOOST_TEST(m == (static_cast<int32_t>(1) << 30));
    BOOST_TEST(s == 0);

    Requantizer<int32_t, int8_t> r;
    r.multiplier = m;
    r.shift = s;
    r.zero_point = 0;
    r.qmin = -128;
    r.qmax = 127;
    BOOST_TEST(static_cast<int>(r.apply(100)) == 50);
}

BOOST_AUTO_TEST_CASE(quantize_multiplier_zero)
{
    int32_t m = 1234;
    int32_t s = 5;
    quantizeMultiplier(0.0, m, s);
    BOOST_TEST(m == 0);
    BOOST_TEST(s == 0);
}

BOOST_AUTO_TEST_CASE(quantize_multiplier_round_trip_small)
{
    // Pick an effective scale ratio < 1 (typical of conv outputs feeding
    // requantization). Verify Requantizer reproduces the float scaling
    // within 1 ULP of the int8 range for a small accumulator.
    const double ratio = 0.125; // exact, no rounding error
    int32_t m = 0;
    int32_t s = 0;
    quantizeMultiplier(ratio, m, s);

    Requantizer<int32_t, int8_t> r;
    r.multiplier = m;
    r.shift = s;
    r.zero_point = 0;
    r.qmin = -128;
    r.qmax = 127;

    BOOST_TEST(static_cast<int>(r.apply(80)) == 10);   // 80 * 0.125
    BOOST_TEST(static_cast<int>(r.apply(800)) == 100); // 800 * 0.125
    BOOST_TEST(static_cast<int>(r.apply(-80)) == -10);
}

// ---------------------------------------------------------------------------
// Phase 2: calibration helpers (qcalibration.hpp).
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(range_observer_tracks_min_max)
{
    RangeObserver obs;
    BOOST_TEST(!obs.has_data);

    obs.observe(0.5f);
    BOOST_TEST(obs.has_data);
    BOOST_TEST(obs.min_value == 0.5f);
    BOOST_TEST(obs.max_value == 0.5f);

    const float buf[] = {-2.0f, 3.5f, 1.0f, -2.5f, 0.0f};
    obs.observe(buf, 5);
    BOOST_TEST(obs.min_value == -2.5f);
    BOOST_TEST(obs.max_value == 3.5f);

    obs.reset();
    BOOST_TEST(!obs.has_data);
    BOOST_TEST(obs.min_value == 0.0f);
}

BOOST_AUTO_TEST_CASE(asymmetric_params_symmetric_input)
{
    // Symmetric input range [-1, 1] over destination [-128, 127]:
    //   scale = 2 / 255, zero_point ~ -1/(2/255) - 128 ~ -0.5, lround -> 0.
    AffineParams p = computeAffineParamsAsymmetric(-1.0f, 1.0f, -128, 127);
    BOOST_TEST(std::abs(p.scale - (2.0f / 255.0f)) < 1e-7f);
    // zp = qmin - fmin/scale = -128 - (-1.0)/(2/255) = -128 + 127.5 = -0.5
    // lround(-0.5) is implementation-defined ties; clamp keeps it in range.
    BOOST_TEST((p.zero_point == 0 || p.zero_point == -1));
}

BOOST_AUTO_TEST_CASE(asymmetric_params_relu_range)
{
    // Post-ReLU activation: range [0, 6], extended by the calibrator to
    // ensure 0 is representable. fmin already 0, fmax=6.
    //   scale = 6/255, zero_point = qmin - 0/scale = -128.
    AffineParams p = computeAffineParamsAsymmetric(0.0f, 6.0f, -128, 127);
    BOOST_TEST(std::abs(p.scale - (6.0f / 255.0f)) < 1e-7f);
    BOOST_TEST(p.zero_point == -128);
}

BOOST_AUTO_TEST_CASE(asymmetric_params_extends_strictly_positive_range)
{
    // Range [2, 4] would skip 0; calibrator must extend fmin down to 0.
    AffineParams p = computeAffineParamsAsymmetric(2.0f, 4.0f, -128, 127);
    BOOST_TEST(std::abs(p.scale - (4.0f / 255.0f)) < 1e-7f);
    BOOST_TEST(p.zero_point == -128);
}

BOOST_AUTO_TEST_CASE(asymmetric_params_zero_range_safe)
{
    AffineParams p = computeAffineParamsAsymmetric(0.0f, 0.0f, -128, 127);
    BOOST_TEST(p.scale == 1.0f);
    BOOST_TEST(p.zero_point == -128);
}

BOOST_AUTO_TEST_CASE(symmetric_params_basic)
{
    // Weights with fmin=-2, fmax=1: absmax=2, qmax_signed=127.
    // scale = 2/127, zero_point = 0.
    AffineParams p = computeAffineParamsSymmetric(-2.0f, 1.0f, 127);
    BOOST_TEST(std::abs(p.scale - (2.0f / 127.0f)) < 1e-7f);
    BOOST_TEST(p.zero_point == 0);
}

BOOST_AUTO_TEST_CASE(symmetric_params_zero_range_safe)
{
    AffineParams p = computeAffineParamsSymmetric(0.0f, 0.0f, 127);
    BOOST_TEST(p.scale == 1.0f);
    BOOST_TEST(p.zero_point == 0);
}

BOOST_AUTO_TEST_CASE(quantize_dequantize_round_trip_within_half_lsb)
{
    // For any value in the calibrated range, |x - dequant(quant(x))| should
    // be at most 0.5 * scale (rounding to nearest grid point).
    const float fmin = -3.0f;
    const float fmax = 3.0f;
    AffineParams p = computeAffineParamsAsymmetric(fmin, fmax, -128, 127);

    const float samples[] = {-3.0f, -1.5f, -0.7f, 0.0f, 0.8f, 1.5f, 3.0f};
    for (float x : samples)
    {
        int8_t q = quantize<int8_t>(x, p.scale, p.zero_point, -128, 127);
        float r = dequantize<int8_t>(q, p.scale, p.zero_point);
        BOOST_TEST(std::abs(r - x) <= 0.5f * p.scale + 1e-6f);
    }
}

BOOST_AUTO_TEST_CASE(quantize_saturates_outside_range)
{
    AffineParams p = computeAffineParamsAsymmetric(-1.0f, 1.0f, -128, 127);
    int8_t hi = quantize<int8_t>(10.0f, p.scale, p.zero_point, -128, 127);
    int8_t lo = quantize<int8_t>(-10.0f, p.scale, p.zero_point, -128, 127);
    BOOST_TEST(static_cast<int>(hi) == 127);
    BOOST_TEST(static_cast<int>(lo) == -128);
}

BOOST_AUTO_TEST_CASE(quantize_buffer_round_trips)
{
    AffineParams p = computeAffineParamsSymmetric(-1.0f, 1.0f, 127);
    const float src[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int8_t qs[5] = {0};
    float roundtrip[5] = {0};
    quantizeBuffer<int8_t>(src, qs, 5, p.scale, p.zero_point, -127, 127);
    dequantizeBuffer<int8_t>(qs, roundtrip, 5, p.scale, p.zero_point);

    for (std::size_t i = 0; i < 5; ++i)
    {
        BOOST_TEST(std::abs(roundtrip[i] - src[i]) <= 0.5f * p.scale + 1e-6f);
    }
}

BOOST_AUTO_TEST_CASE(per_channel_symmetric_scales)
{
    // 3 channels, 4 weights each, with distinct absmaxes 0.5, 1.0, 4.0.
    const float weights[] = {
        0.1f, -0.5f, 0.3f, 0.0f,    // channel 0: absmax 0.5
        -1.0f, 0.5f, 0.0f, 0.9f,    // channel 1: absmax 1.0
        -4.0f, 2.0f, -3.5f, 1.0f,   // channel 2: absmax 4.0
    };
    float scales[3] = {0.0f, 0.0f, 0.0f};
    computePerChannelSymmetricScales(weights, 3, 4, 127, scales);

    BOOST_TEST(std::abs(scales[0] - (0.5f / 127.0f)) < 1e-7f);
    BOOST_TEST(std::abs(scales[1] - (1.0f / 127.0f)) < 1e-7f);
    BOOST_TEST(std::abs(scales[2] - (4.0f / 127.0f)) < 1e-7f);
}

BOOST_AUTO_TEST_CASE(per_channel_zero_channel_falls_back_to_unit)
{
    const float weights[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scales[1] = {0.0f};
    computePerChannelSymmetricScales(weights, 1, 4, 127, scales);
    BOOST_TEST(scales[0] == 1.0f);
}

BOOST_AUTO_TEST_CASE(build_requantizer_matches_manual_decomposition)
{
    // effective_scale = (in * w) / out = (0.25 * 0.5) / 0.25 = 0.5.
    auto r = buildRequantizer<int8_t>(0.25f, 0.5f, 0.25f,
                                      /*output_zp=*/0, -128, 127);
    BOOST_TEST(static_cast<int>(r.apply(100)) == 50);
    BOOST_TEST(static_cast<int>(r.apply(-100)) == -50);

    // Sanity: zero_point passes through.
    auto r2 = buildRequantizer<int8_t>(0.25f, 0.5f, 0.25f,
                                       /*output_zp=*/-10, -128, 127);
    BOOST_TEST(static_cast<int>(r2.apply(100)) == 40);
}
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

// ---------------------------------------------------------------------------
// Phase 3: quantized activations + dense layer. The pure-integer helpers
// (qrelu, qrelu6, clampForRelu*) compile in any configuration; the layer
// tests below them rely on calibration helpers and so are gated again.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(qrelu_clamps_at_zero_point)
{
    // Symmetric int8: zero_point = 0. Negative inputs -> 0, non-negative
    // pass through.
    BOOST_TEST(static_cast<int>(qrelu<int8_t>(-50, 0)) == 0);
    BOOST_TEST(static_cast<int>(qrelu<int8_t>(0, 0)) == 0);
    BOOST_TEST(static_cast<int>(qrelu<int8_t>(50, 0)) == 50);

    // Non-zero zero_point (asymmetric): clamp at zp.
    BOOST_TEST(static_cast<int>(qrelu<int8_t>(-10, -20)) == -10); // > -20
    BOOST_TEST(static_cast<int>(qrelu<int8_t>(-30, -20)) == -20); // < -20
}

BOOST_AUTO_TEST_CASE(qrelu_buffer_in_place)
{
    int8_t buf[5] = {-50, -1, 0, 1, 50};
    qreluBuffer<int8_t>(buf, 5, 0);
    BOOST_TEST(static_cast<int>(buf[0]) == 0);
    BOOST_TEST(static_cast<int>(buf[1]) == 0);
    BOOST_TEST(static_cast<int>(buf[2]) == 0);
    BOOST_TEST(static_cast<int>(buf[3]) == 1);
    BOOST_TEST(static_cast<int>(buf[4]) == 50);
}

BOOST_AUTO_TEST_CASE(qrelu6_clamps_at_both_ends)
{
    // zero_point = -128, q_six = 30 (arbitrary upper clamp).
    BOOST_TEST(static_cast<int>(qrelu6<int8_t>(-128, -128, 30)) == -128);
    BOOST_TEST(static_cast<int>(qrelu6<int8_t>(0, -128, 30)) == 0);
    BOOST_TEST(static_cast<int>(qrelu6<int8_t>(30, -128, 30)) == 30);
    BOOST_TEST(static_cast<int>(qrelu6<int8_t>(50, -128, 30)) == 30);  // saturate high
    BOOST_TEST(static_cast<int>(qrelu6<int8_t>(-127, -128, 30)) == -127); // pass through
    BOOST_TEST(static_cast<int>(qrelu6<int8_t>(-129 + 1, -128, 30)) == -128); // saturate low
}

BOOST_AUTO_TEST_CASE(qrelu6_buffer)
{
    int8_t buf[4] = {-50, 0, 25, 100};
    qrelu6Buffer<int8_t>(buf, 4, 0, 30);
    BOOST_TEST(static_cast<int>(buf[0]) == 0);
    BOOST_TEST(static_cast<int>(buf[1]) == 0);
    BOOST_TEST(static_cast<int>(buf[2]) == 25);
    BOOST_TEST(static_cast<int>(buf[3]) == 30);
}

BOOST_AUTO_TEST_CASE(clamp_for_relu_raises_qmin_only_when_below_zp)
{
    int8_t qmin = -128;
    clampForRelu<int8_t>(0, qmin, 127);
    BOOST_TEST(static_cast<int>(qmin) == 0);

    qmin = 10; // already above zp
    clampForRelu<int8_t>(0, qmin, 127);
    BOOST_TEST(static_cast<int>(qmin) == 10);
}

BOOST_AUTO_TEST_CASE(clamp_for_relu6_lowers_qmax_only_when_above_q_six)
{
    int8_t qmin = -128;
    int8_t qmax = 127;
    clampForRelu6<int8_t>(0, 30, qmin, qmax);
    BOOST_TEST(static_cast<int>(qmin) == 0);
    BOOST_TEST(static_cast<int>(qmax) == 30);

    // q_six above current qmax: qmax should not move up.
    qmin = -128;
    qmax = 20;
    clampForRelu6<int8_t>(0, 30, qmin, qmax);
    BOOST_TEST(static_cast<int>(qmin) == 0);
    BOOST_TEST(static_cast<int>(qmax) == 20);
}

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
BOOST_AUTO_TEST_CASE(quantized_six_threshold_round_trip)
{
    // Output scale 6/255, zero_point = -128. Quantized 6.0 should land at
    // qmax (127).
    const float scale = 6.0f / 255.0f;
    const int32_t zp = -128;
    const int32_t q_six = computeQuantizedSix(scale, zp);
    BOOST_TEST(q_six == 127);

    // Threshold 3.0 lands roughly halfway: qmin + 127 ish.
    const int32_t q_three = computeQuantizedThreshold(3.0f, scale, zp);
    BOOST_TEST(q_three >= -1);
    BOOST_TEST(q_three <= 1);
}

BOOST_AUTO_TEST_CASE(qdense_basic_single_output)
{
    // 2 inputs -> 1 output, weights [1, 1]. Treat all scales as 1.0 so
    // requant ratio is 1.0 and the integer accumulator passes through.
    const int8_t weights[] = {1, 1};
    const int32_t bias[] = {0};

    QDense<int8_t, int8_t, int32_t, int8_t, 2, 1> layer;
    layer.weights = weights;
    layer.biases = bias;
    layer.input_zero_point = 0;
    layer.requantizer = buildRequantizer<int8_t>(
        /*in*/1.0f, /*w*/1.0f, /*out*/1.0f,
        /*out_zp*/0, -128, 127);

    const int8_t input[] = {3, 5};
    int8_t out[1] = {0};
    layer.forward(input, out);
    BOOST_TEST(static_cast<int>(out[0]) == 8);
}

BOOST_AUTO_TEST_CASE(qdense_multi_output)
{
    // 2 inputs -> 2 outputs.
    // weights row-major: [[2, 1],
    //                     [-1, 3]]
    // input = [1, 2]; expect [2*1 + 1*2, -1*1 + 3*2] = [4, 5].
    const int8_t weights[] = {2, 1, -1, 3};
    const int32_t bias[] = {0, 0};

    QDense<int8_t, int8_t, int32_t, int8_t, 2, 2> layer;
    layer.weights = weights;
    layer.biases = bias;
    layer.input_zero_point = 0;
    layer.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    const int8_t input[] = {1, 2};
    int8_t out[2] = {0, 0};
    layer.forward(input, out);
    BOOST_TEST(static_cast<int>(out[0]) == 4);
    BOOST_TEST(static_cast<int>(out[1]) == 5);
}

BOOST_AUTO_TEST_CASE(qdense_applies_bias)
{
    const int8_t weights[] = {1, 1};
    const int32_t bias[] = {10};

    QDense<int8_t, int8_t, int32_t, int8_t, 2, 1> layer;
    layer.weights = weights;
    layer.biases = bias;
    layer.input_zero_point = 0;
    layer.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    const int8_t input[] = {3, 5};
    int8_t out[1] = {0};
    layer.forward(input, out);
    BOOST_TEST(static_cast<int>(out[0]) == 18); // 3+5+10
}

BOOST_AUTO_TEST_CASE(qdense_null_bias_starts_at_zero)
{
    const int8_t weights[] = {2};
    QDense<int8_t, int8_t, int32_t, int8_t, 1, 1> layer;
    layer.weights = weights;
    layer.biases = nullptr;
    layer.input_zero_point = 0;
    layer.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    const int8_t input[] = {7};
    int8_t out[1] = {0};
    layer.forward(input, out);
    BOOST_TEST(static_cast<int>(out[0]) == 14);
}

BOOST_AUTO_TEST_CASE(qdense_subtracts_input_zero_point)
{
    // input_zero_point = 5. Weights [1, 1]. Effective input = (q - 5).
    // q=[10, 20] -> effective [5, 15] -> sum 20.
    const int8_t weights[] = {1, 1};
    const int32_t bias[] = {0};

    QDense<int8_t, int8_t, int32_t, int8_t, 2, 1> layer;
    layer.weights = weights;
    layer.biases = bias;
    layer.input_zero_point = 5;
    layer.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    const int8_t input[] = {10, 20};
    int8_t out[1] = {0};
    layer.forward(input, out);
    BOOST_TEST(static_cast<int>(out[0]) == 20);
}

BOOST_AUTO_TEST_CASE(qdense_with_fused_relu)
{
    // ReLU fused via raised qmin: -128 -> 0. Negative accumulators clamp.
    const int8_t weights[] = {1, -1};
    const int32_t bias[] = {0};

    QDense<int8_t, int8_t, int32_t, int8_t, 2, 1> layer;
    layer.weights = weights;
    layer.biases = bias;
    layer.input_zero_point = 0;
    layer.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);
    int8_t qmin = layer.requantizer.qmin;
    clampForRelu<int8_t>(layer.requantizer.zero_point, qmin, layer.requantizer.qmax);
    layer.requantizer.qmin = qmin;

    // input [5, 10] -> 5 - 10 = -5 -> ReLU -> 0
    const int8_t neg_input[] = {5, 10};
    int8_t out[1] = {0};
    layer.forward(neg_input, out);
    BOOST_TEST(static_cast<int>(out[0]) == 0);

    // input [10, 5] -> 10 - 5 = 5 -> pass through
    const int8_t pos_input[] = {10, 5};
    layer.forward(pos_input, out);
    BOOST_TEST(static_cast<int>(out[0]) == 5);
}

BOOST_AUTO_TEST_CASE(qdense_end_to_end_against_float_reference)
{
    // 3 inputs -> 2 outputs. Build a small float reference, calibrate,
    // quantize weights/inputs, run QDense, dequantize, compare.
    using tinymind::AffineParams;
    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::computeAffineParamsSymmetric;
    using tinymind::quantize;
    using tinymind::dequantize;
    using tinymind::quantizeBuffer;

    const float w_floats[6] = {
         0.5f, -0.25f,  0.75f,   // output 0
        -0.5f,  0.25f, -0.75f,   // output 1
    };
    const float b_floats[2] = {0.1f, -0.1f};
    const float a_floats[3] = {1.0f, -2.0f, 0.5f};

    // Float reference.
    float ref[2] = {0.0f, 0.0f};
    for (int o = 0; o < 2; ++o)
    {
        ref[o] = b_floats[o];
        for (int i = 0; i < 3; ++i)
        {
            ref[o] += w_floats[o * 3 + i] * a_floats[i];
        }
    }

    // Calibrate.
    AffineParams wp = computeAffineParamsSymmetric(-0.75f, 0.75f, 127);
    AffineParams ap = computeAffineParamsAsymmetric(-2.0f, 1.0f, -128, 127);
    AffineParams op = computeAffineParamsAsymmetric(-2.0f, 2.0f, -128, 127);

    // Quantize.
    int8_t wq[6];
    int8_t aq[3];
    int32_t bq[2];
    quantizeBuffer<int8_t>(w_floats, wq, 6, wp.scale, 0, -127, 127);
    quantizeBuffer<int8_t>(a_floats, aq, 3, ap.scale, ap.zero_point, -128, 127);
    // Bias scale = input_scale * weight_scale, no zero_point.
    const float bias_scale = ap.scale * wp.scale;
    for (int o = 0; o < 2; ++o)
    {
        bq[o] = static_cast<int32_t>(std::lround(b_floats[o] / bias_scale));
    }

    QDense<int8_t, int8_t, int32_t, int8_t, 3, 2> layer;
    layer.weights = wq;
    layer.biases = bq;
    layer.input_zero_point = static_cast<int8_t>(ap.zero_point);
    layer.requantizer = buildRequantizer<int8_t>(ap.scale, wp.scale, op.scale,
                                                 op.zero_point, -128, 127);

    int8_t out_q[2];
    layer.forward(aq, out_q);

    float out_f[2];
    out_f[0] = dequantize<int8_t>(out_q[0], op.scale, op.zero_point);
    out_f[1] = dequantize<int8_t>(out_q[1], op.scale, op.zero_point);

    // Allow a few output LSBs of tolerance: weight + activation + bias
    // each contribute their own quantization error.
    BOOST_TEST(std::abs(out_f[0] - ref[0]) <= 4.0f * op.scale);
    BOOST_TEST(std::abs(out_f[1] - ref[1]) <= 4.0f * op.scale);
}
// ---------------------------------------------------------------------------
// Quantized Mixture-of-Experts (top-1 routing).
// ---------------------------------------------------------------------------

namespace
{
    // Configure a QDense expert with identity-pass scales (ratio 1.0) so the
    // int8 output equals the integer accumulator. Caller owns the buffers.
    template<std::size_t In, std::size_t Out>
    void configureExpert(QDense<int8_t, int8_t, int32_t, int8_t, In, Out>& e,
                         const int8_t* weights, const int32_t* biases)
    {
        e.weights = weights;
        e.biases = biases;
        e.input_zero_point = 0;
        e.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);
    }
}

BOOST_AUTO_TEST_CASE(qmoe_static_shape)
{
    typedef QMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 4, 2, 3> MoE;
    static_assert(MoE::InputLength == 4, "Wrong input length");
    static_assert(MoE::OutputLength == 2, "Wrong output length");
    static_assert(MoE::NumberOfExperts == 3, "Wrong expert count");
    static_assert(MoE::Expert::InputLength == 4, "Wrong expert input length");
    static_assert(MoE::Expert::OutputLength == 2, "Wrong expert output length");
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(qmoe_routes_to_argmax_expert_golden)
{
    // 2 inputs -> 1 output, 3 experts. Router logits chosen so expert 1 wins.
    // Expert weights/biases produce a known int8 byte for the golden check.
    typedef QMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 2, 1, 3> MoE;

    // Router rows (row-major [expert][input]). input = {2, 3}, zp = 0:
    //   logit0 = 0*2 + 0*3   = 0
    //   logit1 = 2*2 + 1*3   = 7
    //   logit2 = 1*2 + 0*3   = 2
    const int8_t router_w[6] = {
        0, 0,   // expert 0
        2, 1,   // expert 1
        1, 0,   // expert 2
    };

    // Per-expert weights (each 2 inputs -> 1 output) and biases.
    const int8_t e0_w[2] = {1, 1};  const int32_t e0_b[1] = {0};
    const int8_t e1_w[2] = {3, 0};  const int32_t e1_b[1] = {10};
    const int8_t e2_w[2] = {0, 0};  const int32_t e2_b[1] = {0};

    MoE moe;
    moe.router_weights = router_w;
    moe.router_biases = nullptr;
    moe.input_zero_point = 0;
    configureExpert(moe.experts[0], e0_w, e0_b);
    configureExpert(moe.experts[1], e1_w, e1_b);
    configureExpert(moe.experts[2], e2_w, e2_b);

    const int8_t input[2] = {2, 3};
    int8_t out[1] = {0};
    const std::size_t selected = moe.forward(input, out);

    // argmax(logits {0,7,2}) = expert 1.
    BOOST_TEST(selected == 1u);
    // expert 1 output = 10 + (3*2 + 0*3) = 16  (golden byte)
    BOOST_TEST(static_cast<int>(out[0]) == 16);

    // route() reports the raw int32 logits and the same winner.
    int32_t logits[3] = {0, 0, 0};
    const std::size_t r = moe.route(input, logits);
    BOOST_TEST(r == 1u);
    BOOST_TEST(logits[0] == 0);
    BOOST_TEST(logits[1] == 7);
    BOOST_TEST(logits[2] == 2);
}

BOOST_AUTO_TEST_CASE(qmoe_selection_changes_with_input)
{
    // Two experts; router favors expert 0 on input[0], expert 1 on input[1].
    typedef QMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 2, 1, 2> MoE;

    const int8_t router_w[4] = {
        1, 0,   // expert 0 tracks input[0]
        0, 1,   // expert 1 tracks input[1]
    };
    const int8_t e0_w[2] = {0, 0};  const int32_t e0_b[1] = {-5};
    const int8_t e1_w[2] = {0, 0};  const int32_t e1_b[1] = {5};

    MoE moe;
    moe.router_weights = router_w;
    moe.router_biases = nullptr;
    moe.input_zero_point = 0;
    configureExpert(moe.experts[0], e0_w, e0_b);
    configureExpert(moe.experts[1], e1_w, e1_b);

    int8_t out[1] = {0};

    const int8_t inA[2] = {9, 1};
    BOOST_TEST(moe.forward(inA, out) == 0u);
    BOOST_TEST(static_cast<int>(out[0]) == -5);

    const int8_t inB[2] = {1, 9};
    BOOST_TEST(moe.forward(inB, out) == 1u);
    BOOST_TEST(static_cast<int>(out[0]) == 5);
}

BOOST_AUTO_TEST_CASE(qmoe_argmax_ties_lowest_index)
{
    // All router weights zero with no bias -> all logits 0 (tie) -> expert 0.
    typedef QMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 1, 1, 3> MoE;

    const int8_t router_w[3] = {0, 0, 0};
    const int8_t e0_w[1] = {0};  const int32_t e0_b[1] = {7};
    const int8_t e1_w[1] = {0};  const int32_t e1_b[1] = {8};
    const int8_t e2_w[1] = {0};  const int32_t e2_b[1] = {9};

    MoE moe;
    moe.router_weights = router_w;
    moe.router_biases = nullptr;
    moe.input_zero_point = 0;
    configureExpert(moe.experts[0], e0_w, e0_b);
    configureExpert(moe.experts[1], e1_w, e1_b);
    configureExpert(moe.experts[2], e2_w, e2_b);

    const int8_t input[1] = {5};
    int8_t out[1] = {0};
    BOOST_TEST(moe.forward(input, out) == 0u);
    BOOST_TEST(static_cast<int>(out[0]) == 7);
}

BOOST_AUTO_TEST_CASE(qmoe_router_bias_and_input_zero_point)
{
    // Router uses a bias and an asymmetric input zero point.
    typedef QMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 2, 1, 2> MoE;

    const int8_t router_w[4] = {
        1, 0,
        0, 1,
    };
    // bias tips expert 1 ahead unless input strongly favors expert 0.
    const int32_t router_b[2] = {0, 4};

    const int8_t e0_w[2] = {0, 0};  const int32_t e0_b[1] = {1};
    const int8_t e1_w[2] = {0, 0};  const int32_t e1_b[1] = {2};

    MoE moe;
    moe.router_weights = router_w;
    moe.router_biases = router_b;
    moe.input_zero_point = 5;   // effective input = q - 5
    configureExpert(moe.experts[0], e0_w, e0_b);
    configureExpert(moe.experts[1], e1_w, e1_b);

    // input q = {10, 6} -> effective {5, 1}
    //   logit0 = 0 + 1*5 = 5
    //   logit1 = 4 + 1*1 = 5  -> tie -> expert 0
    const int8_t inTie[2] = {10, 6};
    int8_t out[1] = {0};
    BOOST_TEST(moe.forward(inTie, out) == 0u);
    BOOST_TEST(static_cast<int>(out[0]) == 1);

    // input q = {6, 10} -> effective {1, 5}
    //   logit0 = 1 ; logit1 = 4 + 5 = 9 -> expert 1
    const int8_t inE1[2] = {6, 10};
    BOOST_TEST(moe.forward(inE1, out) == 1u);
    BOOST_TEST(static_cast<int>(out[0]) == 2);
}

// ---------------------------------------------------------------------------
// Quantized top-k / dense Mixture-of-Experts (softmax-gated blend).
// ---------------------------------------------------------------------------

namespace
{
    // Identity-pass requantizer (ratio 1.0, zp 0) reused for router + experts.
    tinymind::Requantizer<int32_t, int8_t> identityRequant()
    {
        return buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);
    }
}

BOOST_AUTO_TEST_CASE(qtopk_static_shape)
{
    typedef QTopKMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 4, 2, 3, 2> MoE;
    static_assert(MoE::InputLength == 4, "Wrong input length");
    static_assert(MoE::OutputLength == 2, "Wrong output length");
    static_assert(MoE::NumberOfExperts == 3, "Wrong expert count");
    static_assert(MoE::TopK == 2, "Wrong K");
    BOOST_TEST(true);
}

BOOST_AUTO_TEST_CASE(qtopk_k1_matches_top1)
{
    // K=1: the gate is 1.0, so the blended output must equal the single
    // argmax expert -- byte-identical to QMixtureOfExperts.
    const int8_t router_w[6] = {
        0, 0,   // expert 0
        2, 1,   // expert 1 (wins for input {2,3})
        1, 0,   // expert 2
    };
    const int8_t e0_w[2] = {1, 1};  const int32_t e0_b[1] = {0};
    const int8_t e1_w[2] = {3, 0};  const int32_t e1_b[1] = {10};
    const int8_t e2_w[2] = {0, 0};  const int32_t e2_b[1] = {0};

    int32_t lut[256];
    tinymind::buildQSoftmaxExpLUT(1.0f, lut);

    QTopKMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 2, 1, 3, 1> moe;
    moe.router_weights = router_w;
    moe.router_biases = nullptr;
    moe.input_zero_point = 0;
    moe.router_requantizer = identityRequant();
    moe.exp_lut = lut;
    moe.output_zero_point = 0;
    moe.output_qmin = -128;
    moe.output_qmax = 127;
    configureExpert(moe.experts[0], e0_w, e0_b);
    configureExpert(moe.experts[1], e1_w, e1_b);
    configureExpert(moe.experts[2], e2_w, e2_b);

    const int8_t input[2] = {2, 3};
    int8_t out[1] = {0};
    const std::size_t top = moe.forward(input, out);

    BOOST_TEST(top == 1u);
    BOOST_TEST(static_cast<int>(out[0]) == 16); // 10 + 3*2 ; same as QMoE top-1
}

BOOST_AUTO_TEST_CASE(qtopk_dense_tied_average_golden)
{
    // Dense MoE (K==NumExperts) with tied router logits -> equal gates ->
    // the blend is the average of the two expert outputs.
    const int8_t router_w[4] = {0, 0, 0, 0};        // logits {0,0} tie
    const int8_t e0_w[2] = {0, 0};  const int32_t e0_b[1] = {10};
    const int8_t e1_w[2] = {0, 0};  const int32_t e1_b[1] = {40};

    int32_t lut[256];
    tinymind::buildQSoftmaxExpLUT(1.0f, lut);

    QTopKMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 2, 1, 2, 2> moe;
    moe.router_weights = router_w;
    moe.router_biases = nullptr;
    moe.input_zero_point = 0;
    moe.router_requantizer = identityRequant();
    moe.exp_lut = lut;
    moe.output_zero_point = 0;
    moe.output_qmin = -128;
    moe.output_qmax = 127;
    configureExpert(moe.experts[0], e0_w, e0_b);
    configureExpert(moe.experts[1], e1_w, e1_b);

    const int8_t input[2] = {5, 7};
    int8_t out[1] = {0};
    moe.forward(input, out);

    // gates 0.5/0.5 over outputs {10,40} -> 25 (golden byte).
    BOOST_TEST(static_cast<int>(out[0]) == 25);
}

BOOST_AUTO_TEST_CASE(qtopk_top2_of_3_excludes_lowest)
{
    // Top-2 of 3: the two highest-logit experts blend; the lowest is dropped.
    const int8_t router_w[3] = {0, 2, 1};   // logits {0,2,1} for input {1}
    const int8_t e0_w[1] = {0};  const int32_t e0_b[1] = {120}; // would saturate if used
    const int8_t e1_w[1] = {0};  const int32_t e1_b[1] = {10};
    const int8_t e2_w[1] = {0};  const int32_t e2_b[1] = {20};

    int32_t lut[256];
    tinymind::buildQSoftmaxExpLUT(1.0f, lut);

    QTopKMixtureOfExperts<int8_t, int8_t, int32_t, int8_t, 1, 1, 3, 2> moe;
    moe.router_weights = router_w;
    moe.router_biases = nullptr;
    moe.input_zero_point = 0;
    moe.router_requantizer = identityRequant();
    moe.exp_lut = lut;
    moe.output_zero_point = 0;
    moe.output_qmin = -128;
    moe.output_qmax = 127;
    configureExpert(moe.experts[0], e0_w, e0_b);
    configureExpert(moe.experts[1], e1_w, e1_b);
    configureExpert(moe.experts[2], e2_w, e2_b);

    const int8_t input[1] = {1};
    int8_t out[1] = {0};
    const std::size_t top = moe.forward(input, out);

    // expert 1 (logit 2) is the top; expert 2 (logit 1) blends in; expert 0
    // (logit 0, value 120) is excluded -> output stays a convex blend of
    // {10, 20}, weighted toward 10.
    BOOST_TEST(top == 1u);
    BOOST_TEST(static_cast<int>(out[0]) >= 10);
    BOOST_TEST(static_cast<int>(out[0]) <= 20);
    BOOST_TEST(static_cast<int>(out[0]) < 15); // expert 1 (10) dominates the softmax
}

// ---------------------------------------------------------------------------
// Phase 4: quantized 2D conv + pool layers.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(qmaxpool2d_2x2_stride2)
{
    // 4x4 single-channel input. Pool 2x2 stride 2 -> 2x2 output.
    // Input grid (row-major NHWC, C=1):
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    //  13 14 15 16
    // Max of each 2x2 quadrant: {6, 8, 14, 16}.
    const int8_t in[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    int8_t out[4] = {0};

    QMaxPool2D<int8_t, 4, 4, 1, 2, 2> pool;
    pool.forward(in, out);

    BOOST_TEST(static_cast<int>(out[0]) == 6);
    BOOST_TEST(static_cast<int>(out[1]) == 8);
    BOOST_TEST(static_cast<int>(out[2]) == 14);
    BOOST_TEST(static_cast<int>(out[3]) == 16);
}

BOOST_AUTO_TEST_CASE(qmaxpool2d_multi_channel)
{
    // 2x2 input, 2 channels, 2x2 pool -> 1x1 output, 2 channels.
    // NHWC layout: pixels are interleaved.
    const int8_t in[8] = {
        1, 100,    2, 50,
        3, 25,     4, 75
    };
    int8_t out[2] = {0, 0};

    QMaxPool2D<int8_t, 2, 2, 2, 2, 2> pool;
    pool.forward(in, out);

    BOOST_TEST(static_cast<int>(out[0]) == 4);    // channel 0 max
    BOOST_TEST(static_cast<int>(out[1]) == 100);  // channel 1 max
}

BOOST_AUTO_TEST_CASE(qavgpool2d_2x2_stride2)
{
    const int8_t in[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    int8_t out[4] = {0};

    QAvgPool2D<int8_t, int32_t, 4, 4, 1, 2, 2> pool;
    pool.qmin = -128;
    pool.qmax = 127;
    pool.forward(in, out);

    // Window means: (1+2+5+6)/4 = 3.5 -> 4 (round-half-away),
    //               (3+4+7+8)/4 = 5.5 -> 6,
    //               (9+10+13+14)/4 = 11.5 -> 12,
    //               (11+12+15+16)/4 = 13.5 -> 14.
    BOOST_TEST(static_cast<int>(out[0]) == 4);
    BOOST_TEST(static_cast<int>(out[1]) == 6);
    BOOST_TEST(static_cast<int>(out[2]) == 12);
    BOOST_TEST(static_cast<int>(out[3]) == 14);
}

BOOST_AUTO_TEST_CASE(qavgpool2d_negative_values_round_half_away)
{
    // Window mean of {-1, -2, -2, -3} = -2.0 (exact). Mean of
    // {-1, -2, -2, -2} = -1.75 -> rounds to -2.
    const int8_t in[8] = {
        -1, -2, -2, -3,    // first window (row, col, c flattened)
        -2, -2, -2, -2,    // padding row to keep H >= PoolH; not used
    };
    int8_t out[1] = {0};

    QAvgPool2D<int8_t, int32_t, 2, 4, 1, 2, 4> pool;
    pool.qmin = -128;
    pool.qmax = 127;
    pool.forward(in, out);

    // Whole 2x4 window covers in[0..3] then in[4..7].
    // Sum = -1 -2 -2 -3 -2 -2 -2 -2 = -16. Avg = -16/8 = -2 exact.
    BOOST_TEST(static_cast<int>(out[0]) == -2);
}

BOOST_AUTO_TEST_CASE(qglobalavgpool2d)
{
    // 2x2 input, 2 channels: NHWC = [c0=1, c1=10, c0=3, c1=20,
    //                                c0=5, c1=30, c0=7, c1=40].
    // Channel 0 mean = (1+3+5+7)/4 = 4.0
    // Channel 1 mean = (10+20+30+40)/4 = 25.0
    const int8_t in[8] = {1, 10, 3, 20, 5, 30, 7, 40};
    int8_t out[2] = {0};

    QGlobalAvgPool2D<int8_t, int32_t, 2, 2, 2> gap;
    gap.qmin = -128;
    gap.qmax = 127;
    gap.forward(in, out);

    BOOST_TEST(static_cast<int>(out[0]) == 4);
    BOOST_TEST(static_cast<int>(out[1]) == 25);
}

BOOST_AUTO_TEST_CASE(qavgpool2d_clamps_to_qmax)
{
    // All values at int8 max -> avg = 127, well within range. Then
    // sanity: tighter qmax forces clamp.
    int8_t in[4] = {127, 127, 127, 127};
    int8_t out[1] = {0};

    QAvgPool2D<int8_t, int32_t, 2, 2, 1, 2, 2> pool;
    pool.qmin = -128;
    pool.qmax = 50;  // artificially tight for test
    pool.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 50);
}

BOOST_AUTO_TEST_CASE(qconv2d_identity_kernel)
{
    // 3x3 input, 3x3 kernel = 1x1 output. Single in/out channel.
    // Kernel is all zeros except center weight = 1; bias 0; identity
    // ratio (in*w/out) = 1.0, zero zp.
    const int8_t in[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const int8_t weights[9] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0,
    };
    const int32_t bias[1] = {0};
    int8_t out[1] = {0};

    QConv2D<int8_t, int8_t, int32_t, int8_t, 3, 3, 1, 3, 3, 1, 1, 1> conv;
    conv.weights = weights;
    conv.biases = bias;
    conv.input_zero_point = 0;
    conv.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    conv.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 5); // center value
}

BOOST_AUTO_TEST_CASE(qconv2d_two_filters)
{
    // 3x3 input, 3x3 kernel, 2 filters -> 1x1x2 output.
    // Filter 0: identity (center=1). Filter 1: -identity.
    const int8_t in[9] = {
        1, 2, 3,
        4, 10, 6,
        7, 8, 9,
    };
    const int8_t weights[18] = {
        // filter 0 (KH*KW*Cin = 9 weights)
        0, 0, 0, 0, 1, 0, 0, 0, 0,
        // filter 1
        0, 0, 0, 0, -1, 0, 0, 0, 0,
    };
    const int32_t bias[2] = {0, 0};
    int8_t out[2] = {0};

    QConv2D<int8_t, int8_t, int32_t, int8_t, 3, 3, 1, 3, 3, 1, 1, 2> conv;
    conv.weights = weights;
    conv.biases = bias;
    conv.input_zero_point = 0;
    conv.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    conv.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 10);   // center
    BOOST_TEST(static_cast<int>(out[1]) == -10);  // negated center
}

BOOST_AUTO_TEST_CASE(qconv2d_with_bias_and_input_zp)
{
    // 2x2 input, 2x2 kernel, 1 filter -> 1x1 out.
    // input_zero_point = 5; effective input = q - 5.
    // Weights all 1 -> effective sum = (q1-5) + (q2-5) + (q3-5) + (q4-5).
    const int8_t in[4] = {10, 10, 10, 10};   // effective 5 + 5 + 5 + 5 = 20
    const int8_t weights[4] = {1, 1, 1, 1};
    const int32_t bias[1] = {3};
    int8_t out[1] = {0};

    QConv2D<int8_t, int8_t, int32_t, int8_t, 2, 2, 1, 2, 2, 1, 1, 1> conv;
    conv.weights = weights;
    conv.biases = bias;
    conv.input_zero_point = 5;
    conv.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    conv.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 23); // 20 + 3
}

BOOST_AUTO_TEST_CASE(qconv2d_stride_two_2x4_input)
{
    // 2x4 input, 1x2 kernel, stride W=2 -> 2x2 output (height H=2 with
    // KH=1 stride=1 -> 2; width (4-2)/2+1 = 2).
    // Wait — with KH=1 we'd need H>=1; let's use H=2, KH=2, strideH=1
    // -> OH = (2-2)/1 + 1 = 1. Actually pick KH=1 strideH=1 H=2 -> OH=2.
    // Use KH=1, KW=2, strideH=1, strideW=2, H=2, W=4 -> OH=2, OW=2.
    const int8_t in[8] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    const int8_t weights[2] = {1, 1}; // KH*KW*Cin = 2
    const int32_t bias[1] = {0};
    int8_t out[4] = {0};

    QConv2D<int8_t, int8_t, int32_t, int8_t, 2, 4, 1, 1, 2, 1, 2, 1> conv;
    conv.weights = weights;
    conv.biases = bias;
    conv.input_zero_point = 0;
    conv.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    conv.forward(in, out);
    // (1+2), (3+4), (5+6), (7+8) -> 3, 7, 11, 15.
    BOOST_TEST(static_cast<int>(out[0]) == 3);
    BOOST_TEST(static_cast<int>(out[1]) == 7);
    BOOST_TEST(static_cast<int>(out[2]) == 11);
    BOOST_TEST(static_cast<int>(out[3]) == 15);
}

BOOST_AUTO_TEST_CASE(qconv2d_end_to_end_against_float_reference)
{
    // 4x4 input single channel, 3x3 kernel, 1 filter, stride 1.
    // Output is 2x2. Compare quantized output (dequantized) to float.
    using tinymind::AffineParams;

    const float a_floats[16] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 0.5f, -0.5f,
        -0.1f, -0.2f, -0.3f, -0.4f,
    };
    const float w_floats[9] = {
        0.1f, 0.2f, 0.1f,
        0.0f, 0.5f, 0.0f,
        -0.1f, -0.2f, -0.1f,
    };
    const float b_floats[1] = {0.05f};

    // Float reference.
    float ref[4] = {0};
    for (int oh = 0; oh < 2; ++oh)
    {
        for (int ow = 0; ow < 2; ++ow)
        {
            float acc = b_floats[0];
            for (int kh = 0; kh < 3; ++kh)
            {
                for (int kw = 0; kw < 3; ++kw)
                {
                    acc += w_floats[kh * 3 + kw] *
                           a_floats[(oh + kh) * 4 + (ow + kw)];
                }
            }
            ref[oh * 2 + ow] = acc;
        }
    }

    AffineParams ap = computeAffineParamsAsymmetric(-0.5f, 1.0f, -128, 127);
    AffineParams wp = computeAffineParamsSymmetric(-0.5f, 0.5f, 127);
    AffineParams op = computeAffineParamsAsymmetric(-1.5f, 1.5f, -128, 127);

    int8_t aq[16];
    int8_t wq[9];
    int32_t bq[1];
    quantizeBuffer<int8_t>(a_floats, aq, 16, ap.scale, ap.zero_point, -128, 127);
    quantizeBuffer<int8_t>(w_floats, wq, 9, wp.scale, 0, -127, 127);
    const float bias_scale = ap.scale * wp.scale;
    bq[0] = static_cast<int32_t>(std::lround(b_floats[0] / bias_scale));

    QConv2D<int8_t, int8_t, int32_t, int8_t, 4, 4, 1, 3, 3, 1, 1, 1> conv;
    conv.weights = wq;
    conv.biases = bq;
    conv.input_zero_point = static_cast<int8_t>(ap.zero_point);
    conv.requantizer = buildRequantizer<int8_t>(ap.scale, wp.scale, op.scale,
                                                op.zero_point, -128, 127);

    int8_t out_q[4];
    conv.forward(aq, out_q);

    for (int i = 0; i < 4; ++i)
    {
        const float out_f = dequantize<int8_t>(out_q[i], op.scale, op.zero_point);
        // Tolerance: weight + activation + bias quantization combined.
        BOOST_TEST(std::abs(out_f - ref[i]) <= 5.0f * op.scale);
    }
}

// ---------------------------------------------------------------------------
// Phase 5: per-channel depthwise + per-tensor pointwise conv.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(qdepthwiseconv2d_two_channels_distinct_scales)
{
    // 3x3 input, 2 channels, 3x3 kernel, 1 filter per channel -> 1x1x2 out.
    // Channel 0: identity center weight = 1, ratio 1.0.
    // Channel 1: identity center weight = 1, but per-channel ratio 0.5
    // (effectively halves the output) — exercises the per-channel
    // Requantizer array.
    //
    // NHWC: input[h, w, c] interleaved.
    // Channel 0 grid: 1 2 3 / 4 50 6 / 7 8 9. Center = 50.
    // Channel 1 grid: 10 20 30 / 40 100 60 / 70 80 90. Center = 100.
    const int8_t in[18] = {
        1, 10,  2, 20,  3, 30,
        4, 40, 50,100,  6, 60,
        7, 70,  8, 80,  9, 90,
    };
    const int8_t weights[18] = {
        // channel 0 KH*KW = 9
        0, 0, 0, 0, 1, 0, 0, 0, 0,
        // channel 1
        0, 0, 0, 0, 1, 0, 0, 0, 0,
    };
    const int32_t bias[2] = {0, 0};

    Requantizer<int32_t, int8_t> rs[2];
    rs[0] = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);
    rs[1] = buildRequantizer<int8_t>(1.0f, 1.0f, 2.0f, 0, -128, 127);

    QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 3, 3, 2, 3, 3, 1, 1> dw;
    dw.weights = weights;
    dw.biases = bias;
    dw.input_zero_point = 0;
    dw.requantizers = rs;

    int8_t out[2] = {0};
    dw.forward(in, out);

    BOOST_TEST(static_cast<int>(out[0]) == 50);  // channel 0: passes through
    BOOST_TEST(static_cast<int>(out[1]) == 50);  // channel 1: 100 * 0.5 = 50
}

BOOST_AUTO_TEST_CASE(qdepthwiseconv2d_with_input_zero_point)
{
    // 2x2 input, 1 channel, 2x2 kernel -> 1x1 out.
    // input_zp = 5; effective input = q - 5.
    // weights all 1, bias 0. q=[10,10,10,10] -> effective sum = 5+5+5+5 = 20.
    const int8_t in[4] = {10, 10, 10, 10};
    const int8_t weights[4] = {1, 1, 1, 1};
    const int32_t bias[1] = {0};

    Requantizer<int32_t, int8_t> rs[1];
    rs[0] = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 2, 2, 1, 2, 2, 1, 1> dw;
    dw.weights = weights;
    dw.biases = bias;
    dw.input_zero_point = 5;
    dw.requantizers = rs;

    int8_t out[1] = {0};
    dw.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 20);
}

BOOST_AUTO_TEST_CASE(qdepthwiseconv2d_null_bias_starts_at_zero)
{
    // Single 1x1 input/kernel, single channel — degenerate but exercises
    // the nullptr-bias path in depthwise.
    const int8_t in[1] = {7};
    const int8_t weights[1] = {2};

    Requantizer<int32_t, int8_t> rs[1];
    rs[0] = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 1, 1, 1, 1, 1, 1, 1> dw;
    dw.weights = weights;
    dw.biases = nullptr;
    dw.input_zero_point = 0;
    dw.requantizers = rs;

    int8_t out[1] = {0};
    dw.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 14);
}

BOOST_AUTO_TEST_CASE(qdepthwiseconv2d_per_channel_calibrated_against_float)
{
    // 3x3 input, 2 channels (very different ranges), 2x2 depthwise kernel.
    // Per-channel symmetric weight scale; per-channel Requantizer.
    using tinymind::AffineParams;

    // Distinct activation/weight ranges per channel.
    // Channel 0 activations small ~[-0.5, 0.5], weights small.
    // Channel 1 activations big ~[-4, 4], weights bigger.
    const float a_floats_c0[9] = {
         0.1f,  0.2f,  0.3f,
         0.4f,  0.5f, -0.5f,
        -0.4f, -0.3f, -0.2f,
    };
    const float a_floats_c1[9] = {
        -3.0f, -2.0f, -1.0f,
         0.0f,  1.0f,  2.0f,
         3.0f,  4.0f, -4.0f,
    };

    const float w_floats_c0[4] = { 0.1f, 0.2f, -0.1f, 0.05f };
    const float w_floats_c1[4] = { 1.5f, -1.0f, 0.5f, -0.5f };

    const float b_floats[2] = { 0.01f, -0.5f };

    // Float reference (NHWC interleaving for input).
    float ref[8] = {0};
    for (int oh = 0; oh < 2; ++oh)
    {
        for (int ow = 0; ow < 2; ++ow)
        {
            for (int c = 0; c < 2; ++c)
            {
                float acc = b_floats[c];
                for (int kh = 0; kh < 2; ++kh)
                {
                    for (int kw = 0; kw < 2; ++kw)
                    {
                        const int ih = oh + kh;
                        const int iw = ow + kw;
                        const float a = (c == 0)
                            ? a_floats_c0[ih * 3 + iw]
                            : a_floats_c1[ih * 3 + iw];
                        const float w = (c == 0)
                            ? w_floats_c0[kh * 2 + kw]
                            : w_floats_c1[kh * 2 + kw];
                        acc += w * a;
                    }
                }
                ref[(oh * 2 + ow) * 2 + c] = acc;
            }
        }
    }

    // Calibrate. Activation is per-tensor across both channels (TFLite
    // typically uses per-tensor activations even when weights are
    // per-channel). Pick a range that covers both channels.
    AffineParams ap = computeAffineParamsAsymmetric(-4.0f, 4.0f, -128, 127);
    AffineParams op = computeAffineParamsAsymmetric(-10.0f, 10.0f, -128, 127);

    // Per-channel symmetric weight params.
    AffineParams wp[2];
    wp[0] = computeAffineParamsSymmetric(-0.1f, 0.2f, 127);
    wp[1] = computeAffineParamsSymmetric(-1.0f, 1.5f, 127);

    // Build interleaved input (NHWC, 2 channels).
    int8_t aq[18];
    for (int i = 0; i < 9; ++i)
    {
        aq[i * 2 + 0] = quantize<int8_t>(a_floats_c0[i],
                                         ap.scale, ap.zero_point, -128, 127);
        aq[i * 2 + 1] = quantize<int8_t>(a_floats_c1[i],
                                         ap.scale, ap.zero_point, -128, 127);
    }

    // Quantize per-channel weights into a single channel-major buffer of
    // shape [Channels][KH*KW].
    int8_t wq[8];
    for (int i = 0; i < 4; ++i)
    {
        wq[0 * 4 + i] = quantize<int8_t>(w_floats_c0[i], wp[0].scale, 0, -127, 127);
        wq[1 * 4 + i] = quantize<int8_t>(w_floats_c1[i], wp[1].scale, 0, -127, 127);
    }

    // Per-channel int32 bias with scale = ap.scale * wp[c].scale.
    int32_t bq[2];
    for (int c = 0; c < 2; ++c)
    {
        const float bias_scale = ap.scale * wp[c].scale;
        bq[c] = static_cast<int32_t>(std::lround(b_floats[c] / bias_scale));
    }

    Requantizer<int32_t, int8_t> rs[2];
    for (int c = 0; c < 2; ++c)
    {
        rs[c] = buildRequantizer<int8_t>(ap.scale, wp[c].scale, op.scale,
                                         op.zero_point, -128, 127);
    }

    QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 3, 3, 2, 2, 2, 1, 1> dw;
    dw.weights = wq;
    dw.biases = bq;
    dw.input_zero_point = static_cast<int8_t>(ap.zero_point);
    dw.requantizers = rs;

    int8_t out_q[8];
    dw.forward(aq, out_q);

    for (int i = 0; i < 8; ++i)
    {
        const float out_f = dequantize<int8_t>(out_q[i], op.scale, op.zero_point);
        BOOST_TEST(std::abs(out_f - ref[i]) <= 6.0f * op.scale);
    }
}

BOOST_AUTO_TEST_CASE(qpointwiseconv2d_basic_reduction)
{
    // 1x1 spatial, 3 input channels, 2 output channels.
    // weights row-major [NumFilters][InChannels] = [[1, 1, 1], [-1, 0, 1]].
    // input = [4, 5, 6]. Expected: [15, 2].
    const int8_t in[3] = {4, 5, 6};
    const int8_t weights[6] = {
         1,  1, 1,
        -1,  0, 1,
    };
    const int32_t bias[2] = {0, 0};

    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, 1, 1, 3, 2> pw;
    pw.weights = weights;
    pw.biases = bias;
    pw.input_zero_point = 0;
    pw.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    int8_t out[2] = {0};
    pw.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 15);
    BOOST_TEST(static_cast<int>(out[1]) == 2);
}

BOOST_AUTO_TEST_CASE(qpointwiseconv2d_2x2_spatial)
{
    // 2x2 spatial, 2 input channels, 1 output channel.
    // weights = [1, 1] (single filter).
    // Each output pixel = c0 + c1 at that location.
    const int8_t in[8] = {
        1, 2,    3, 4,
        5, 6,    7, 8,
    };
    const int8_t weights[2] = {1, 1};
    const int32_t bias[1] = {10};

    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, 2, 2, 2, 1> pw;
    pw.weights = weights;
    pw.biases = bias;
    pw.input_zero_point = 0;
    pw.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    int8_t out[4] = {0};
    pw.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 13);  // 1+2+10
    BOOST_TEST(static_cast<int>(out[1]) == 17);  // 3+4+10
    BOOST_TEST(static_cast<int>(out[2]) == 21);  // 5+6+10
    BOOST_TEST(static_cast<int>(out[3]) == 25);  // 7+8+10
}

BOOST_AUTO_TEST_CASE(qpointwiseconv2d_input_zp_subtracts)
{
    // 1x1 spatial, 2 in channels, 1 filter; weights [1, 1]; input_zp = 3.
    // input = [10, 10] -> effective [7, 7] -> sum 14.
    const int8_t in[2] = {10, 10};
    const int8_t weights[2] = {1, 1};
    const int32_t bias[1] = {0};

    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, 1, 1, 2, 1> pw;
    pw.weights = weights;
    pw.biases = bias;
    pw.input_zero_point = 3;
    pw.requantizer = buildRequantizer<int8_t>(1.0f, 1.0f, 1.0f, 0, -128, 127);

    int8_t out[1] = {0};
    pw.forward(in, out);
    BOOST_TEST(static_cast<int>(out[0]) == 14);
}

BOOST_AUTO_TEST_CASE(end_to_end_quantized_dot_product)
{
    // Verify the full calibration path: float weights + activations get
    // quantized, an int32 dot product is requantized back to int8, then
    // dequantized to a float that's close to the float reference.
    const float w_floats[] = {0.5f, -0.25f, 0.75f, 1.0f};
    const float a_floats[] = {1.0f, 2.0f, -1.0f, 0.5f};
    float ref = 0.0f;
    for (int i = 0; i < 4; ++i) ref += w_floats[i] * a_floats[i];
    // ref = 0.5 - 0.5 - 0.75 + 0.5 = -0.25

    AffineParams wp = computeAffineParamsSymmetric(-0.25f, 1.0f, 127);
    AffineParams ap = computeAffineParamsAsymmetric(-1.0f, 2.0f, -128, 127);
    AffineParams op = computeAffineParamsAsymmetric(-2.0f, 2.0f, -128, 127);

    int8_t wq[4];
    int8_t aq[4];
    quantizeBuffer<int8_t>(w_floats, wq, 4, wp.scale, 0, -127, 127);
    quantizeBuffer<int8_t>(a_floats, aq, 4, ap.scale, ap.zero_point, -128, 127);

    int32_t acc = 0;
    for (int i = 0; i < 4; ++i)
    {
        acc += static_cast<int32_t>(wq[i]) *
               (static_cast<int32_t>(aq[i]) - ap.zero_point);
    }

    auto r = buildRequantizer<int8_t>(ap.scale, wp.scale, op.scale,
                                      op.zero_point, -128, 127);
    int8_t out_q = r.apply(acc);
    float out = dequantize<int8_t>(out_q, op.scale, op.zero_point);

    // 1 LSB of output scale is op.scale; allow a couple LSBs of tolerance
    // since input/weight quantization each contribute their own rounding.
    BOOST_TEST(std::abs(out - ref) <= 3.0f * op.scale);
}
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

BOOST_AUTO_TEST_CASE(qactivation_lut_index_endpoints)
{
    // -128 maps to index 0, 0 maps to index 128, 127 maps to index 255.
    BOOST_TEST(qActivationLUTIndex(static_cast<int8_t>(-128)) == 0u);
    BOOST_TEST(qActivationLUTIndex(static_cast<int8_t>(0)) == 128u);
    BOOST_TEST(qActivationLUTIndex(static_cast<int8_t>(127)) == 255u);
}

BOOST_AUTO_TEST_CASE(qactivation_lut_apply_pointwise_and_buffer)
{
    int8_t lut[kQActivationLUTSize];
    for (std::size_t i = 0; i < kQActivationLUTSize; ++i)
    {
        lut[i] = static_cast<int8_t>(static_cast<int32_t>(i) - 128);
    }

    // Identity LUT: output equals input across the full int8 range.
    for (int32_t v = -128; v <= 127; ++v)
    {
        BOOST_TEST(qApplyLUT(static_cast<int8_t>(v), lut) == static_cast<int8_t>(v));
    }

    int8_t buf[5] = {-128, -1, 0, 1, 127};
    qApplyLUTBuffer(buf, 5, lut);
    BOOST_TEST(buf[0] == static_cast<int8_t>(-128));
    BOOST_TEST(buf[1] == static_cast<int8_t>(-1));
    BOOST_TEST(buf[2] == static_cast<int8_t>(0));
    BOOST_TEST(buf[3] == static_cast<int8_t>(1));
    BOOST_TEST(buf[4] == static_cast<int8_t>(127));
}

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

BOOST_AUTO_TEST_CASE(qsigmoid_lut_matches_float_reference)
{
    // Calibrate the input tensor over [-8, 8] (saturating sigmoid range).
    AffineParams ip = computeAffineParamsAsymmetric(-8.0f, 8.0f, -128, 127);
    // Sigmoid output lives in (0, 1); use the canonical 1/256 grid with
    // zero_point = -128 so the full int8 range covers the function.
    const float out_scale = 1.0f / 256.0f;
    const int32_t out_zp = -128;

    int8_t lut[kQActivationLUTSize];
    buildQSigmoidLUT(ip.scale, ip.zero_point, out_scale, out_zp, lut);

    // The LUT must be monotonic non-decreasing in the input value.
    for (std::size_t i = 1; i < kQActivationLUTSize; ++i)
    {
        BOOST_TEST(static_cast<int32_t>(lut[i]) >= static_cast<int32_t>(lut[i - 1]));
    }

    // Saturation extremes: very negative input -> ~0, very positive -> ~1.
    {
        const int8_t q_neg = quantize<int8_t>(-7.5f, ip.scale, ip.zero_point, -128, 127);
        const int8_t y_neg = qApplyLUT(q_neg, lut);
        const float yf = dequantize<int8_t>(y_neg, out_scale, out_zp);
        BOOST_TEST(yf < 0.01f);
    }
    {
        const int8_t q_pos = quantize<int8_t>(7.5f, ip.scale, ip.zero_point, -128, 127);
        const int8_t y_pos = qApplyLUT(q_pos, lut);
        const float yf = dequantize<int8_t>(y_pos, out_scale, out_zp);
        BOOST_TEST(yf > 0.99f);
    }

    // Sample several points and compare to the float reference. The error
    // budget is the input grid step times sigmoid'(x)<=1/4 plus one output
    // LSB; allow a couple of LSBs for rounding.
    const float test_xs[] = {-4.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 4.0f};
    for (float x : test_xs)
    {
        const int8_t qx = quantize<int8_t>(x, ip.scale, ip.zero_point, -128, 127);
        const float y_q = dequantize<int8_t>(qApplyLUT(qx, lut), out_scale, out_zp);
        const float y_ref = 1.0f / (1.0f + std::exp(-x));
        BOOST_TEST(std::abs(y_q - y_ref) < 0.05f);
    }
}

BOOST_AUTO_TEST_CASE(qtanh_lut_matches_float_reference)
{
    AffineParams ip = computeAffineParamsAsymmetric(-4.0f, 4.0f, -128, 127);
    // tanh output lives in (-1, 1); 1/128 with zp=0 covers the symmetric range.
    const float out_scale = 1.0f / 128.0f;
    const int32_t out_zp = 0;

    int8_t lut[kQActivationLUTSize];
    buildQTanhLUT(ip.scale, ip.zero_point, out_scale, out_zp, lut);

    for (std::size_t i = 1; i < kQActivationLUTSize; ++i)
    {
        BOOST_TEST(static_cast<int32_t>(lut[i]) >= static_cast<int32_t>(lut[i - 1]));
    }

    // Odd symmetry around the input zero_point: tanh(0) = 0.
    {
        const int8_t qx = quantize<int8_t>(0.0f, ip.scale, ip.zero_point, -128, 127);
        const float y_q = dequantize<int8_t>(qApplyLUT(qx, lut), out_scale, out_zp);
        BOOST_TEST(std::abs(y_q) < 0.02f);
    }

    const float test_xs[] = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f};
    for (float x : test_xs)
    {
        const int8_t qx = quantize<int8_t>(x, ip.scale, ip.zero_point, -128, 127);
        const float y_q = dequantize<int8_t>(qApplyLUT(qx, lut), out_scale, out_zp);
        const float y_ref = std::tanh(x);
        BOOST_TEST(std::abs(y_q - y_ref) < 0.05f);
    }
}

// Phase 15 -- calibration upgrades.
BOOST_AUTO_TEST_CASE(percentile_observer_clips_outliers)
{
    PercentileObserver obs;
    // Bulk sample mass in [-1, 1] plus a handful of >> 1 outliers. Naive
    // min/max would set absmax to the outlier value; percentile clipping
    // at 1.0 / 99.0 should land near 1.0.
    for (int i = -1000; i <= 1000; ++i)
    {
        obs.observe(static_cast<float>(i) / 1000.0f);
    }
    const float kOutliers[] = {50.0f, -50.0f, 75.0f, -75.0f};
    obs.observe(kOutliers, sizeof(kOutliers) / sizeof(kOutliers[0]));

    float fmin = 0.0f;
    float fmax = 0.0f;
    obs.rangeAtPercentile(1.0f, 99.0f, fmin, fmax);
    BOOST_TEST(std::abs(fmin - (-0.98f)) < 0.01f);
    BOOST_TEST(std::abs(fmax -   0.98f) < 0.01f);

    // 0.0 / 100.0 returns the true min/max including outliers.
    PercentileObserver obs_full;
    for (int i = -1000; i <= 1000; ++i)
    {
        obs_full.observe(static_cast<float>(i) / 1000.0f);
    }
    obs_full.observe(kOutliers, sizeof(kOutliers) / sizeof(kOutliers[0]));
    float fmin_full = 0.0f;
    float fmax_full = 0.0f;
    obs_full.rangeAtPercentile(0.0f, 100.0f, fmin_full, fmax_full);
    BOOST_TEST(fmin_full <= -50.0f);
    BOOST_TEST(fmax_full >=  50.0f);
}

BOOST_AUTO_TEST_CASE(percentile_observer_empty_returns_zero_range)
{
    PercentileObserver obs;
    float fmin = -42.0f;
    float fmax =  42.0f;
    obs.rangeAtPercentile(0.0f, 100.0f, fmin, fmax);
    BOOST_TEST(fmin == 0.0f);
    BOOST_TEST(fmax == 0.0f);
    BOOST_TEST(obs.has_data == false);
}

BOOST_AUTO_TEST_CASE(kl_divergence_observer_finds_clip_threshold)
{
    // Build a Gaussian-ish distribution with heavy tail outliers. KL
    // calibration should pick a threshold well below the absmax.
    KLDivergenceObserver obs;
    std::vector<float> data;
    data.reserve(20000 + 50);
    for (int i = 0; i < 20000; ++i)
    {
        const float u = static_cast<float>((i % 1024) - 512) / 512.0f;
        data.push_back(u * 0.7f);
    }
    for (int i = 0; i < 50; ++i)
    {
        data.push_back((i & 1) ? 8.0f : -8.0f);
    }
    obs.observeAbsRange(data.data(), data.size());
    obs.observeHistogram(data.data(), data.size());
    const float threshold = obs.computeThreshold();
    BOOST_TEST(obs.absmax >= 7.5f);
    BOOST_TEST(threshold > 0.0f);
    BOOST_TEST(threshold <= obs.absmax);
    // Clip threshold should be substantially below the outlier absmax.
    BOOST_TEST(threshold < obs.absmax * 0.5f);
}

BOOST_AUTO_TEST_CASE(kl_divergence_observer_empty_returns_zero)
{
    KLDivergenceObserver obs;
    BOOST_TEST(obs.computeThreshold() == 0.0f);
    BOOST_TEST(obs.has_data == false);
}

BOOST_AUTO_TEST_CASE(cross_layer_equalize_dense_preserves_relu_output)
{
    // Two-layer ReLU MLP. CLE rescales W1/b1/W2 by per-channel s[c];
    // because ReLU is positively homogeneous, model output must be
    // bitwise unchanged in the float domain (up to numerical noise).
    constexpr std::size_t In = 4;
    constexpr std::size_t Mid = 6;
    constexpr std::size_t Out = 3;
    float w1[Mid * In];
    float b1[Mid];
    float w2[Out * Mid];
    // Engineered imbalance: channel 0 of mid has tiny upstream weights
    // but huge downstream weights; channel 1 is the opposite.
    for (std::size_t c = 0; c < Mid; ++c)
    {
        const float scale_up   = (c == 0) ? 0.01f : (c == 1) ? 10.0f : 1.0f;
        const float scale_down = (c == 0) ? 10.0f : (c == 1) ? 0.01f : 1.0f;
        for (std::size_t i = 0; i < In; ++i)
        {
            w1[c * In + i] = scale_up * (0.5f + 0.1f * static_cast<float>(i + c));
        }
        b1[c] = scale_up * 0.25f;
        for (std::size_t o = 0; o < Out; ++o)
        {
            w2[o * Mid + c] = scale_down *
                              (0.3f + 0.07f * static_cast<float>(o + c));
        }
    }

    const float x[In] = {0.7f, -0.2f, 0.5f, 0.1f};

    auto forward = [&](const float* W1, const float* B1, const float* W2,
                       float* y) {
        float h[Mid];
        for (std::size_t c = 0; c < Mid; ++c)
        {
            float acc = B1[c];
            for (std::size_t i = 0; i < In; ++i)
            {
                acc += W1[c * In + i] * x[i];
            }
            h[c] = (acc > 0.0f) ? acc : 0.0f;
        }
        for (std::size_t o = 0; o < Out; ++o)
        {
            float acc = 0.0f;
            for (std::size_t c = 0; c < Mid; ++c)
            {
                acc += W2[o * Mid + c] * h[c];
            }
            y[o] = acc;
        }
    };

    float y_before[Out];
    forward(w1, b1, w2, y_before);

    // Record per-channel ranges before / after to confirm equalization.
    float r1_before[Mid];
    float r2_before[Mid];
    for (std::size_t c = 0; c < Mid; ++c)
    {
        float r1 = 0.0f;
        for (std::size_t i = 0; i < In; ++i)
        {
            const float a = std::abs(w1[c * In + i]);
            if (a > r1) r1 = a;
        }
        float r2 = 0.0f;
        for (std::size_t o = 0; o < Out; ++o)
        {
            const float a = std::abs(w2[o * Mid + c]);
            if (a > r2) r2 = a;
        }
        r1_before[c] = r1;
        r2_before[c] = r2;
    }

    crossLayerEqualizeDense(w1, b1, w2, In, Mid, Out);

    float y_after[Out];
    forward(w1, b1, w2, y_after);
    for (std::size_t o = 0; o < Out; ++o)
    {
        BOOST_TEST(std::abs(y_after[o] - y_before[o]) < 1.0e-4f);
    }

    // Channels 0 and 1 had ratios 0.01/10 and 10/0.01 = 1e6; after CLE
    // the per-channel r1/r2 ratio should be close to 1.
    for (std::size_t c = 0; c < 2; ++c)
    {
        float r1 = 0.0f;
        float r2 = 0.0f;
        for (std::size_t i = 0; i < In; ++i)
        {
            const float a = std::abs(w1[c * In + i]);
            if (a > r1) r1 = a;
        }
        for (std::size_t o = 0; o < Out; ++o)
        {
            const float a = std::abs(w2[o * Mid + c]);
            if (a > r2) r2 = a;
        }
        const float ratio_after = r1 / r2;
        const float ratio_before = r1_before[c] / r2_before[c];
        BOOST_TEST(std::abs(ratio_after - 1.0f) < 0.05f);
        // Original imbalance must have been at least 50x in either
        // direction; CLE pulls it back to near 1.0.
        const float log_before = std::log(ratio_before);
        BOOST_TEST(std::abs(log_before) > 4.0f);
    }
}

BOOST_AUTO_TEST_CASE(cross_layer_equalize_dense_skips_zero_channels)
{
    // Channel 0 of W1 is identically zero. CLE must leave that channel
    // alone (no division by zero, output still preserved).
    constexpr std::size_t In = 2;
    constexpr std::size_t Mid = 3;
    constexpr std::size_t Out = 2;
    float w1[Mid * In] = {
        0.0f, 0.0f,
        1.0f, 0.5f,
        -0.3f, 0.7f,
    };
    float b1[Mid] = {0.1f, 0.2f, -0.05f};
    float w2[Out * Mid] = {
        0.4f,  0.8f, -0.6f,
        -0.2f, 0.9f,  0.1f,
    };

    const float x[In] = {0.5f, -0.25f};
    auto forward = [&](const float* W1, const float* B1, const float* W2,
                       float* y) {
        float h[Mid];
        for (std::size_t c = 0; c < Mid; ++c)
        {
            float acc = B1[c];
            for (std::size_t i = 0; i < In; ++i) acc += W1[c * In + i] * x[i];
            h[c] = (acc > 0.0f) ? acc : 0.0f;
        }
        for (std::size_t o = 0; o < Out; ++o)
        {
            float acc = 0.0f;
            for (std::size_t c = 0; c < Mid; ++c)
                acc += W2[o * Mid + c] * h[c];
            y[o] = acc;
        }
    };

    float y_before[Out];
    forward(w1, b1, w2, y_before);
    crossLayerEqualizeDense(w1, b1, w2, In, Mid, Out);
    float y_after[Out];
    forward(w1, b1, w2, y_after);
    for (std::size_t o = 0; o < Out; ++o)
    {
        BOOST_TEST(std::abs(y_after[o] - y_before[o]) < 1.0e-5f);
    }
    BOOST_TEST(w1[0] == 0.0f);
    BOOST_TEST(w1[1] == 0.0f);
    BOOST_TEST(b1[0] == 0.1f);
}

BOOST_AUTO_TEST_CASE(cross_layer_equalize_conv2d_preserves_relu_output)
{
    // Conv -> ReLU -> Conv. CLE on per-output-filter channel of the first
    // conv, matching input channel of the second. Output unchanged.
    constexpr std::size_t F1 = 3;   // filters of conv1
    constexpr std::size_t Kh1 = 2;
    constexpr std::size_t Kw1 = 2;
    constexpr std::size_t IC1 = 1;  // input channels to conv1
    constexpr std::size_t WPF1 = Kh1 * Kw1 * IC1;
    constexpr std::size_t F2 = 2;
    constexpr std::size_t Kh2 = 1;
    constexpr std::size_t Kw2 = 1;
    constexpr std::size_t WPF2 = Kh2 * Kw2 * F1;
    // Input image 3x3x1
    constexpr std::size_t H = 3;
    constexpr std::size_t W = 3;
    float w1[F1 * WPF1];
    float b1[F1];
    float w2[F2 * WPF2];

    for (std::size_t f = 0; f < F1; ++f)
    {
        const float s_up = (f == 0) ? 0.01f : (f == 1) ? 5.0f : 1.0f;
        for (std::size_t k = 0; k < WPF1; ++k)
        {
            w1[f * WPF1 + k] = s_up * (0.2f + 0.05f * static_cast<float>(k));
        }
        b1[f] = s_up * 0.1f;
    }
    for (std::size_t f = 0; f < F2; ++f)
    {
        for (std::size_t c = 0; c < F1; ++c)
        {
            const float s_dn = (c == 0) ? 5.0f : (c == 1) ? 0.01f : 1.0f;
            w2[f * WPF2 + c] = s_dn * (0.3f + 0.1f * static_cast<float>(f + c));
        }
    }

    float input[H * W * IC1];
    for (std::size_t i = 0; i < H * W; ++i)
    {
        input[i] = static_cast<float>(i) * 0.1f - 0.4f;
    }

    auto conv_forward = [&](const float* W1, const float* B1,
                            const float* W2_, float* y) {
        // conv1: 3x3 input, 2x2 VALID -> 2x2 output, F1 channels.
        // y2 shape: 2 x 2 x F2.
        float mid[(H - 1) * (W - 1) * F1];
        for (std::size_t oh = 0; oh < H - 1; ++oh)
        {
            for (std::size_t ow = 0; ow < W - 1; ++ow)
            {
                for (std::size_t f = 0; f < F1; ++f)
                {
                    float acc = B1[f];
                    for (std::size_t kh = 0; kh < Kh1; ++kh)
                    {
                        for (std::size_t kw = 0; kw < Kw1; ++kw)
                        {
                            const std::size_t in_idx =
                                (oh + kh) * W + (ow + kw);
                            const std::size_t w_idx =
                                f * WPF1 + (kh * Kw1 + kw) * IC1;
                            acc += W1[w_idx] * input[in_idx];
                        }
                    }
                    mid[(oh * (W - 1) + ow) * F1 + f] = (acc > 0.0f) ? acc : 0.0f;
                }
            }
        }
        // conv2: 1x1 over 2x2 mid -> 2x2 x F2.
        for (std::size_t oh = 0; oh < H - 1; ++oh)
        {
            for (std::size_t ow = 0; ow < W - 1; ++ow)
            {
                for (std::size_t f = 0; f < F2; ++f)
                {
                    float acc = 0.0f;
                    for (std::size_t c = 0; c < F1; ++c)
                    {
                        acc += W2_[f * WPF2 + c] *
                               mid[(oh * (W - 1) + ow) * F1 + c];
                    }
                    y[(oh * (W - 1) + ow) * F2 + f] = acc;
                }
            }
        }
    };

    float y_before[(H - 1) * (W - 1) * F2];
    conv_forward(w1, b1, w2, y_before);

    crossLayerEqualizeConv2D(w1, b1, w2, F1, WPF1, F2, Kh2, Kw2);

    float y_after[(H - 1) * (W - 1) * F2];
    conv_forward(w1, b1, w2, y_after);
    for (std::size_t i = 0; i < (H - 1) * (W - 1) * F2; ++i)
    {
        BOOST_TEST(std::abs(y_after[i] - y_before[i]) < 1.0e-4f);
    }
}

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

// ---------------------------------------------------------------------------
// Phase 9: type bridges + fp16/bf16 storage tier.
//
// Bridge tests use the Q8.8 fixed-point type as the canonical Q-format
// representative. Round-trip tolerance reflects the unavoidable single
// rounding step at the boundary (Q8.8 has ~3.9e-3 resolution; int8
// affine over a 1.0 range has ~7.8e-3 resolution).
// ---------------------------------------------------------------------------

#if TINYMIND_ENABLE_FLOAT

using tinymind::affineDequantize;
using tinymind::affineQuantize;
using tinymind::qValueToFloat;
using tinymind::floatToQValue;
using tinymind::qValueToAffine;
using tinymind::affineToQValue;
using tinymind::qValueToFloatBuffer;
using tinymind::floatToQValueBuffer;
using tinymind::qValueToAffineBuffer;
using tinymind::affineToQValueBuffer;

typedef tinymind::QValue<8, 8, true> Q88Bridge;

BOOST_AUTO_TEST_CASE(qbridge_affine_dequant_quant_roundtrip)
{
    // scale 0.05, zero_point -10 -> covers about [-5.9, 6.85] in float
    const float scale = 0.05f;
    const int32_t zp = -10;

    const float xs[] = {-5.0f, -1.25f, 0.0f, 0.25f, 3.5f, 6.0f};
    for (float x : xs)
    {
        const int8_t q = affineQuantize<int8_t>(x, scale, zp, -128, 127);
        const float r = affineDequantize<int8_t>(q, scale, zp);
        BOOST_TEST(std::abs(r - x) < scale);  // within one quant step
    }
}

BOOST_AUTO_TEST_CASE(qbridge_qvalue_float_roundtrip)
{
    // Q8.8: ~3.9e-3 resolution, range about [-128, 127.996].
    const float xs[] = {-100.0f, -1.5f, -0.125f, 0.0f, 0.125f, 1.5f, 100.0f};
    for (float x : xs)
    {
        const Q88Bridge qv = floatToQValue<Q88Bridge>(x);
        const float r = qValueToFloat<Q88Bridge>(qv);
        // 1 LSB tolerance.
        const float lsb = 1.0f / static_cast<float>(1 << Q88Bridge::NumberOfFractionalBits);
        BOOST_TEST(std::abs(r - x) <= lsb);
    }
}

BOOST_AUTO_TEST_CASE(qbridge_qvalue_to_affine_roundtrip)
{
    // Pipeline: float -> QValue -> int8 affine -> float (decode).
    const float scale = 0.04f;
    const int32_t zp = 5;
    const float xs[] = {-3.0f, -0.75f, 0.0f, 0.5f, 2.5f};
    for (float x : xs)
    {
        const Q88Bridge qv = floatToQValue<Q88Bridge>(x);
        const int8_t q = qValueToAffine<Q88Bridge, int8_t>(qv, scale, zp, -128, 127);
        const float r = affineDequantize<int8_t>(q, scale, zp);
        // Two rounding steps: Q-format LSB plus affine LSB.
        const float qlsb = 1.0f / static_cast<float>(1 << Q88Bridge::NumberOfFractionalBits);
        BOOST_TEST(std::abs(r - x) < scale + qlsb);
    }
}

BOOST_AUTO_TEST_CASE(qbridge_affine_to_qvalue_roundtrip)
{
    // Pipeline: float -> int8 affine -> QValue -> float.
    const float scale = 0.04f;
    const int32_t zp = 5;
    const float xs[] = {-3.0f, -0.75f, 0.0f, 0.5f, 2.5f};
    for (float x : xs)
    {
        const int8_t q = affineQuantize<int8_t>(x, scale, zp, -128, 127);
        const Q88Bridge qv = affineToQValue<Q88Bridge, int8_t>(q, scale, zp);
        const float r = qValueToFloat<Q88Bridge>(qv);
        const float qlsb = 1.0f / static_cast<float>(1 << Q88Bridge::NumberOfFractionalBits);
        BOOST_TEST(std::abs(r - x) < scale + qlsb);
    }
}

BOOST_AUTO_TEST_CASE(qbridge_affine_quantize_saturates)
{
    const float scale = 0.5f;
    const int32_t zp = 0;

    BOOST_TEST(affineQuantize<int8_t>(1000.0f, scale, zp, -128, 127) == 127);
    BOOST_TEST(affineQuantize<int8_t>(-1000.0f, scale, zp, -128, 127) == -128);
    BOOST_TEST(affineQuantize<int8_t>(0.0f, scale, zp, -128, 127) == 0);
}

BOOST_AUTO_TEST_CASE(qbridge_buffer_round_trip)
{
    const float scale = 0.02f;
    const int32_t zp = -3;
    const float src[] = {-2.0f, -0.5f, 0.0f, 0.25f, 1.75f};
    constexpr std::size_t n = sizeof(src) / sizeof(float);

    Q88Bridge qbuf[n];
    int8_t i8buf[n];
    Q88Bridge round_qbuf[n];
    float round_fbuf[n];

    floatToQValueBuffer<Q88Bridge>(src, qbuf, n);
    qValueToAffineBuffer<Q88Bridge, int8_t>(qbuf, i8buf, n, scale, zp, -128, 127);
    affineToQValueBuffer<Q88Bridge, int8_t>(i8buf, round_qbuf, n, scale, zp);
    qValueToFloatBuffer<Q88Bridge>(round_qbuf, round_fbuf, n);

    const float qlsb = 1.0f / static_cast<float>(1 << Q88Bridge::NumberOfFractionalBits);
    for (std::size_t i = 0; i < n; ++i)
    {
        BOOST_TEST(std::abs(round_fbuf[i] - src[i]) < scale + qlsb);
    }
}

// Phase 17: pure-integer Q <-> int8 bridge. Same round-trip semantics as
// the float-mediated bridge above, but uses the (multiplier, shift) Q0.31
// primitive instead of an internal float. Built host-side, consumed by the
// deployable freestanding shape (FLOAT=0 STD=0 QUANT=1).

using tinymind::AffineToQValueIntParams;
using tinymind::QValueToAffineIntParams;
using tinymind::affineToQValueInt;
using tinymind::qValueToAffineInt;
using tinymind::buildAffineToQValueIntParams;
using tinymind::buildQValueToAffineIntParams;
using tinymind::affineToQValueIntBuffer;
using tinymind::qValueToAffineIntBuffer;

BOOST_AUTO_TEST_CASE(qbridge_int_affine_to_qvalue_matches_float_bridge)
{
    const float scale = 0.04f;
    const int32_t zp = 5;
    const AffineToQValueIntParams<Q88Bridge> params =
        buildAffineToQValueIntParams<Q88Bridge>(scale, zp);

    for (int32_t qi = -64; qi <= 64; qi += 8)
    {
        const int8_t q = static_cast<int8_t>(qi);
        const Q88Bridge fbridge = affineToQValue<Q88Bridge, int8_t>(q, scale, zp);
        const Q88Bridge ibridge = affineToQValueInt<Q88Bridge, int8_t>(q, params);
        // Integer bridge uses gemmlowp Q0.31 rounding; float bridge uses
        // round-half-away-from-zero on a float. Allow 1 LSB drift.
        const int64_t diff = static_cast<int64_t>(ibridge.getValue())
                           - static_cast<int64_t>(fbridge.getValue());
        BOOST_TEST((diff >= -1 && diff <= 1));
    }
}

BOOST_AUTO_TEST_CASE(qbridge_int_qvalue_to_affine_matches_float_bridge)
{
    const float scale = 0.04f;
    const int32_t zp = 5;
    const QValueToAffineIntParams<Q88Bridge> params =
        buildQValueToAffineIntParams<Q88Bridge>(scale, zp, -128, 127);

    const float xs[] = {-3.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 3.0f};
    for (float x : xs)
    {
        const Q88Bridge qv = floatToQValue<Q88Bridge>(x);
        const int8_t fbridge = qValueToAffine<Q88Bridge, int8_t>(qv, scale, zp,
                                                                 -128, 127);
        const int8_t ibridge = qValueToAffineInt<Q88Bridge, int8_t>(qv, params);
        const int32_t diff = static_cast<int32_t>(ibridge)
                           - static_cast<int32_t>(fbridge);
        BOOST_TEST((diff >= -1 && diff <= 1));
    }
}

BOOST_AUTO_TEST_CASE(qbridge_int_round_trip_within_tolerance)
{
    // float -> QValue -> int8 (integer bridge) -> QValue (integer bridge) ->
    // float must close back within one affine LSB plus one QValue LSB.
    const float scale = 0.04f;
    const int32_t zp = -3;
    const QValueToAffineIntParams<Q88Bridge> q2a =
        buildQValueToAffineIntParams<Q88Bridge>(scale, zp, -128, 127);
    const AffineToQValueIntParams<Q88Bridge> a2q =
        buildAffineToQValueIntParams<Q88Bridge>(scale, zp);

    const float xs[] = {-2.5f, -0.5f, 0.0f, 0.5f, 2.5f};
    for (float x : xs)
    {
        const Q88Bridge qv_in  = floatToQValue<Q88Bridge>(x);
        const int8_t   q       = qValueToAffineInt<Q88Bridge, int8_t>(qv_in, q2a);
        const Q88Bridge qv_out = affineToQValueInt<Q88Bridge, int8_t>(q, a2q);
        const float    r       = qValueToFloat<Q88Bridge>(qv_out);
        const float    qlsb    = 1.0f /
            static_cast<float>(1 << Q88Bridge::NumberOfFractionalBits);
        BOOST_TEST(std::abs(r - x) < scale + qlsb);
    }
}

BOOST_AUTO_TEST_CASE(qbridge_int_qvalue_to_affine_saturates)
{
    // scale tight enough that any reasonable QValue saturates int8 range.
    const float scale = 0.001f;
    const int32_t zp = 0;
    const QValueToAffineIntParams<Q88Bridge> p =
        buildQValueToAffineIntParams<Q88Bridge>(scale, zp, -128, 127);

    const Q88Bridge big = floatToQValue<Q88Bridge>(100.0f);
    const Q88Bridge sm  = floatToQValue<Q88Bridge>(-100.0f);
    BOOST_TEST((qValueToAffineInt<Q88Bridge, int8_t>(big, p) == 127));
    BOOST_TEST((qValueToAffineInt<Q88Bridge, int8_t>(sm,  p) == -128));
}

BOOST_AUTO_TEST_CASE(qbridge_int_buffer_round_trip)
{
    const float scale = 0.02f;
    const int32_t zp = -3;
    const QValueToAffineIntParams<Q88Bridge> q2a =
        buildQValueToAffineIntParams<Q88Bridge>(scale, zp, -128, 127);
    const AffineToQValueIntParams<Q88Bridge> a2q =
        buildAffineToQValueIntParams<Q88Bridge>(scale, zp);

    const float src[] = {-2.0f, -0.5f, 0.0f, 0.25f, 1.75f};
    constexpr std::size_t n = sizeof(src) / sizeof(float);

    Q88Bridge qbuf[n];
    int8_t i8buf[n];
    Q88Bridge round_qbuf[n];
    float round_fbuf[n];

    floatToQValueBuffer<Q88Bridge>(src, qbuf, n);
    qValueToAffineIntBuffer<Q88Bridge, int8_t>(qbuf, i8buf, n, q2a);
    affineToQValueIntBuffer<Q88Bridge, int8_t>(i8buf, round_qbuf, n, a2q);
    qValueToFloatBuffer<Q88Bridge>(round_qbuf, round_fbuf, n);

    const float qlsb = 1.0f / static_cast<float>(1 << Q88Bridge::NumberOfFractionalBits);
    for (std::size_t i = 0; i < n; ++i)
    {
        BOOST_TEST(std::abs(round_fbuf[i] - src[i]) < scale + qlsb);
    }
}

#if TINYMIND_ENABLE_FP16

using tinymind::fp16_t;
using tinymind::bf16_t;
using tinymind::floatToFp16;
using tinymind::fp16ToFloat;
using tinymind::floatToBf16;
using tinymind::bf16ToFloat;
using tinymind::fp16ToAffineI8;
using tinymind::affineI8ToFp16;
using tinymind::bf16ToAffineI8;
using tinymind::affineI8ToBf16;

BOOST_AUTO_TEST_CASE(fp16_round_trip_normals)
{
    const float xs[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
                        1.234f, -42.0f, 1024.0f, -7.5f};
    for (float x : xs)
    {
        const fp16_t h = floatToFp16(x);
        const float r = fp16ToFloat(h);
        // fp16 normal: 11-bit mantissa precision -> relative error <= 2^-10.
        const float tol = 1e-3f * (1.0f + std::abs(x));
        BOOST_TEST(std::abs(r - x) < tol);
    }
}

BOOST_AUTO_TEST_CASE(fp16_zero_inf_nan)
{
    BOOST_TEST(fp16ToFloat(floatToFp16(0.0f)) == 0.0f);
    BOOST_TEST(fp16ToFloat(floatToFp16(-0.0f)) == 0.0f);

    const fp16_t pinf = floatToFp16(1.0e30f);  // overflow
    BOOST_TEST(std::isinf(fp16ToFloat(pinf)));
    BOOST_TEST(fp16ToFloat(pinf) > 0.0f);

    const fp16_t ninf = floatToFp16(-1.0e30f);
    BOOST_TEST(std::isinf(fp16ToFloat(ninf)));
    BOOST_TEST(fp16ToFloat(ninf) < 0.0f);
}

BOOST_AUTO_TEST_CASE(bf16_round_trip)
{
    const float xs[] = {0.0f, 1.0f, -1.0f, 1024.0f, -42.0f, 1.234f, 1e-20f};
    for (float x : xs)
    {
        const bf16_t b = floatToBf16(x);
        const float r = bf16ToFloat(b);
        // bf16: 8-bit mantissa precision -> relative error <= 2^-7.
        const float tol = 8e-3f * (1.0f + std::abs(x));
        BOOST_TEST(std::abs(r - x) < tol);
    }
}

BOOST_AUTO_TEST_CASE(bf16_preserves_fp32_exponent_range)
{
    // bf16 shares fp32's exponent field, so very large and very small
    // values that fp16 would overflow / underflow stay representable.
    const float big = 1.0e30f;
    const float small = 1.0e-30f;
    BOOST_TEST(std::isfinite(bf16ToFloat(floatToBf16(big))));
    BOOST_TEST(bf16ToFloat(floatToBf16(small)) > 0.0f);
}

BOOST_AUTO_TEST_CASE(bridge_fp16_to_int8_affine_round_trip)
{
    const float scale = 0.05f;
    const int32_t zp = 0;
    const float xs[] = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    for (float x : xs)
    {
        const fp16_t h = floatToFp16(x);
        const int8_t q = fp16ToAffineI8(h, scale, zp, -128, 127);
        const fp16_t back = affineI8ToFp16(q, scale, zp);
        const float r = fp16ToFloat(back);
        BOOST_TEST(std::abs(r - x) < scale + 1e-2f);
    }
}

BOOST_AUTO_TEST_CASE(bridge_bf16_to_int8_affine_round_trip)
{
    const float scale = 0.05f;
    const int32_t zp = 0;
    const float xs[] = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    for (float x : xs)
    {
        const bf16_t b = floatToBf16(x);
        const int8_t q = bf16ToAffineI8(b, scale, zp, -128, 127);
        const bf16_t back = affineI8ToBf16(q, scale, zp);
        const float r = bf16ToFloat(back);
        BOOST_TEST(std::abs(r - x) < scale + 5e-2f);
    }
}

BOOST_AUTO_TEST_CASE(fp16_subnormals_round_trip)
{
    // Smallest fp16 normal is 2^-14 (~6.10e-5). Values below that land in
    // the subnormal range down to 2^-24 (~5.96e-8): exercises floatToFp16's
    // subnormal branch (implicit-1 shift + round) and fp16ToFloat's
    // renormalize loop.
    const float xs[] = {6.0e-5f, 3.0e-5f, 1.0e-5f, 5.0e-6f, 1.0e-6f,
                        1.0e-7f, -3.0e-5f, -1.0e-6f};
    for (float x : xs)
    {
        const fp16_t h = floatToFp16(x);
        const float r = fp16ToFloat(h);
        // Subnormal step is 2^-24; allow one step of absolute error.
        BOOST_TEST(std::abs(r - x) <= 6.0e-8f + 1e-9f);
        BOOST_TEST((r > 0.0f) == (x > 0.0f));
    }
}

BOOST_AUTO_TEST_CASE(fp16_underflow_to_zero)
{
    // Magnitude below half the smallest subnormal (2^-25) rounds to a
    // signed zero: exercises the new_exp < -10 early return.
    BOOST_TEST(fp16ToFloat(floatToFp16(1.0e-10f)) == 0.0f);
    BOOST_TEST(fp16ToFloat(floatToFp16(-1.0e-10f)) == 0.0f);
}

BOOST_AUTO_TEST_CASE(fp16_nan_input_preserved)
{
    // exp_f == 0xFF with non-zero mantissa: NaN in, NaN out (sets the
    // fp16 quiet-NaN mantissa bit, decoded back through the exp==0x1F path).
    BOOST_TEST(std::isnan(fp16ToFloat(floatToFp16(NAN))));
}

BOOST_AUTO_TEST_CASE(fp16_true_infinity_input)
{
    // exp_f == 0xFF with zero mantissa (genuine Inf input, distinct from
    // the finite-overflow path the existing test covers).
    const fp16_t pinf = floatToFp16(INFINITY);
    const fp16_t ninf = floatToFp16(-INFINITY);
    BOOST_TEST(std::isinf(fp16ToFloat(pinf)));
    BOOST_TEST(fp16ToFloat(pinf) > 0.0f);
    BOOST_TEST(std::isinf(fp16ToFloat(ninf)));
    BOOST_TEST(fp16ToFloat(ninf) < 0.0f);
}

BOOST_AUTO_TEST_CASE(fp16_rounding_carries_into_exponent)
{
    // 65535.0 sits just above the largest finite fp16 (65504): rounding the
    // mantissa carries past 0x3FF, bumps the exponent to 0x1F, and snaps to
    // Inf -- exercises the mantissa-overflow / exponent-bump branch.
    BOOST_TEST(std::isinf(fp16ToFloat(floatToFp16(65535.0f))));

    // A dense normal-range sweep also walks the ties-to-even carry path that
    // bumps the mantissa without overflowing the exponent.
    for (int i = 1; i < 4000; ++i)
    {
        const float x = static_cast<float>(i) * 0.017f;
        const float r = fp16ToFloat(floatToFp16(x));
        BOOST_TEST(std::abs(r - x) < 1e-2f * (1.0f + std::abs(x)));
    }
}

BOOST_AUTO_TEST_CASE(bf16_nan_and_inf_preserved)
{
    // exp_field == 0xFF, mantissa != 0: NaN preservation branch.
    BOOST_TEST(std::isnan(bf16ToFloat(floatToBf16(NAN))));
    // Genuine Inf flows through the round-to-nearest path unchanged.
    BOOST_TEST(std::isinf(bf16ToFloat(floatToBf16(INFINITY))));
    BOOST_TEST(bf16ToFloat(floatToBf16(INFINITY)) > 0.0f);
    BOOST_TEST(bf16ToFloat(floatToBf16(-INFINITY)) < 0.0f);
}

BOOST_AUTO_TEST_CASE(fp16_bf16_buffer_round_trip)
{
    const float src[] = {0.0f, 1.0f, -2.5f, 42.0f, -0.125f, 1024.0f};
    const std::size_t n = sizeof(src) / sizeof(src[0]);

    fp16_t hbuf[n];
    float hback[n];
    tinymind::floatToFp16Buffer(src, hbuf, n);
    tinymind::fp16ToFloatBuffer(hbuf, hback, n);

    bf16_t bbuf[n];
    float bback[n];
    tinymind::floatToBf16Buffer(src, bbuf, n);
    tinymind::bf16ToFloatBuffer(bbuf, bback, n);

    for (std::size_t i = 0; i < n; ++i)
    {
        const float tol = 1.0f + std::abs(src[i]);
        BOOST_TEST(std::abs(hback[i] - src[i]) < 1e-2f * tol);
        BOOST_TEST(std::abs(bback[i] - src[i]) < 1e-1f * tol);
    }
}

#endif // TINYMIND_ENABLE_FP16

#endif // TINYMIND_ENABLE_FLOAT

// ---------------------------------------------------------------------------
// Phase 10: composition ops (QAdd, QMul, QConcat2_2D, QPad2D) + per-channel
// QConv2D / QPointwiseConv2D.
//
// Parity against a float reference: quantize inputs, run the int8 op, then
// dequantize the output and compare against the same op done in float.
// Tolerances reflect the destination quant step plus one bit of slack.
// ---------------------------------------------------------------------------

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

BOOST_AUTO_TEST_CASE(qadd_parity_with_float_reference)
{
    const float scale_a = 0.05f;
    const int32_t zp_a = -5;
    const float scale_b = 0.07f;
    const int32_t zp_b = 10;
    const float scale_y = 0.10f;
    const int32_t zp_y = 0;

    const float a_vals[] = {-1.5f, 0.0f, 1.0f, 2.5f, -2.0f};
    const float b_vals[] = {0.5f, -0.25f, 1.5f, -1.0f, 2.0f};
    constexpr std::size_t N = sizeof(a_vals) / sizeof(float);

    int8_t qa[N], qb[N], qy[N];
    quantizeBuffer<int8_t>(a_vals, qa, N, scale_a, zp_a, -128, 127);
    quantizeBuffer<int8_t>(b_vals, qb, N, scale_b, zp_b, -128, 127);

    QAddParams p = buildQAddParams(scale_a, scale_b, scale_y);
    QAdd<int8_t, int8_t, int8_t, N> add;
    add.input_a_zero_point = static_cast<int8_t>(zp_a);
    add.input_b_zero_point = static_cast<int8_t>(zp_b);
    add.left_shift = p.left_shift;
    add.input_a_multiplier = p.input_a_multiplier;
    add.input_a_shift = p.input_a_shift;
    add.input_b_multiplier = p.input_b_multiplier;
    add.input_b_shift = p.input_b_shift;
    add.output_requantizer.multiplier = p.output_multiplier;
    add.output_requantizer.shift = p.output_shift;
    add.output_requantizer.zero_point = static_cast<int8_t>(zp_y);
    add.output_requantizer.qmin = -128;
    add.output_requantizer.qmax = 127;

    add.forward(qa, qb, qy);

    for (std::size_t i = 0; i < N; ++i)
    {
        const float deq = dequantize<int8_t>(qy[i], scale_y, zp_y);
        const float ref = a_vals[i] + b_vals[i];
        BOOST_TEST(std::abs(deq - ref) < scale_y + 1e-3f);
    }
}

BOOST_AUTO_TEST_CASE(qadd_saturation)
{
    // Inputs designed to drive the output above qmax.
    const float scale = 0.5f;
    QAddParams p = buildQAddParams(scale, scale, scale);

    QAdd<int8_t, int8_t, int8_t, 1> add;
    add.input_a_zero_point = 0;
    add.input_b_zero_point = 0;
    add.left_shift = p.left_shift;
    add.input_a_multiplier = p.input_a_multiplier;
    add.input_a_shift = p.input_a_shift;
    add.input_b_multiplier = p.input_b_multiplier;
    add.input_b_shift = p.input_b_shift;
    add.output_requantizer.multiplier = p.output_multiplier;
    add.output_requantizer.shift = p.output_shift;
    add.output_requantizer.zero_point = 0;
    add.output_requantizer.qmin = -128;
    add.output_requantizer.qmax = 127;

    int8_t a = 100, b = 100, y = 0;
    add.forward(&a, &b, &y);
    BOOST_TEST(y == 127);

    int8_t na = -100, nb = -100, ny = 0;
    add.forward(&na, &nb, &ny);
    BOOST_TEST(ny == -128);
}

BOOST_AUTO_TEST_CASE(qmul_parity_with_float_reference)
{
    const float scale_a = 0.04f;
    const int32_t zp_a = 0;
    const float scale_b = 0.06f;
    const int32_t zp_b = -3;
    // scale_y picked so the worst-case product magnitude (max|a|*max|b| = 3.0)
    // stays within int8 range without saturation.
    const float scale_y = 0.05f;
    const int32_t zp_y = 5;

    const float a_vals[] = {0.5f, -0.75f, 1.0f, -1.5f, 0.25f};
    const float b_vals[] = {1.0f, 0.5f, -0.5f, -2.0f, 0.75f};
    constexpr std::size_t N = sizeof(a_vals) / sizeof(float);

    int8_t qa[N], qb[N], qy[N];
    quantizeBuffer<int8_t>(a_vals, qa, N, scale_a, zp_a, -128, 127);
    quantizeBuffer<int8_t>(b_vals, qb, N, scale_b, zp_b, -128, 127);

    QMul<int8_t, int8_t, int8_t, N> mul;
    mul.input_a_zero_point = static_cast<int8_t>(zp_a);
    mul.input_b_zero_point = static_cast<int8_t>(zp_b);
    mul.requantizer = buildQMulRequantizer<int8_t>(scale_a, scale_b, scale_y,
                                                   zp_y, -128, 127);
    mul.forward(qa, qb, qy);

    for (std::size_t i = 0; i < N; ++i)
    {
        const float deq = dequantize<int8_t>(qy[i], scale_y, zp_y);
        const float ref = a_vals[i] * b_vals[i];
        // Multiplication compounds the LSB error of both inputs; tolerance
        // is two output-quant steps plus a small float slack.
        BOOST_TEST(std::abs(deq - ref) < 2.0f * scale_y + 1e-2f);
    }
}

BOOST_AUTO_TEST_CASE(qconcat_2d_parity_with_float_reference)
{
    constexpr std::size_t H = 2;
    constexpr std::size_t W = 2;
    constexpr std::size_t CA = 2;
    constexpr std::size_t CB = 3;
    constexpr std::size_t N_A = H * W * CA;
    constexpr std::size_t N_B = H * W * CB;
    constexpr std::size_t N_OUT = H * W * (CA + CB);

    const float scale_a = 0.03f;
    const int32_t zp_a = -10;
    const float scale_b = 0.05f;
    const int32_t zp_b = 5;
    const float scale_y = 0.05f;
    const int32_t zp_y = 0;

    float a_f[N_A], b_f[N_B];
    for (std::size_t i = 0; i < N_A; ++i) a_f[i] = -1.0f + 0.25f * static_cast<float>(i);
    for (std::size_t i = 0; i < N_B; ++i) b_f[i] = 0.5f - 0.2f * static_cast<float>(i);

    int8_t qa[N_A], qb[N_B], qy[N_OUT];
    quantizeBuffer<int8_t>(a_f, qa, N_A, scale_a, zp_a, -128, 127);
    quantizeBuffer<int8_t>(b_f, qb, N_B, scale_b, zp_b, -128, 127);

    QConcat2_2D<int8_t, int8_t, int8_t, H, W, CA, CB> concat;
    concat.input_a_zero_point = static_cast<int8_t>(zp_a);
    concat.input_b_zero_point = static_cast<int8_t>(zp_b);
    concat.requantizer_a = buildRescaler<int8_t>(scale_a, scale_y, zp_y, -128, 127);
    concat.requantizer_b = buildRescaler<int8_t>(scale_b, scale_y, zp_y, -128, 127);
    concat.forward(qa, qb, qy);

    for (std::size_t h = 0; h < H; ++h)
    {
        for (std::size_t w = 0; w < W; ++w)
        {
            const std::size_t out_off = (h * W + w) * (CA + CB);
            const std::size_t a_off = (h * W + w) * CA;
            const std::size_t b_off = (h * W + w) * CB;
            for (std::size_t c = 0; c < CA; ++c)
            {
                const float deq = dequantize<int8_t>(qy[out_off + c], scale_y, zp_y);
                BOOST_TEST(std::abs(deq - a_f[a_off + c]) < scale_y + 1e-2f);
            }
            for (std::size_t c = 0; c < CB; ++c)
            {
                const float deq = dequantize<int8_t>(qy[out_off + CA + c], scale_y, zp_y);
                BOOST_TEST(std::abs(deq - b_f[b_off + c]) < scale_y + 1e-2f);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(qpad_2d_inserts_zero_point_border)
{
    constexpr std::size_t H_IN = 2;
    constexpr std::size_t W_IN = 2;
    constexpr std::size_t C = 2;
    constexpr std::size_t PT = 1, PB = 1, PL = 1, PR = 1;

    QPad2D<int8_t, H_IN, W_IN, C, PT, PB, PL, PR> pad;
    pad.pad_value = static_cast<int8_t>(-7);

    int8_t in[H_IN * W_IN * C];
    for (std::size_t i = 0; i < H_IN * W_IN * C; ++i) in[i] = static_cast<int8_t>(i + 1);

    int8_t out[decltype(pad)::OutputSize];
    pad.forward(in, out);

    // Top row, bottom row, left col, right col -> pad value.
    const std::size_t OH = pad.OutputHeight;
    const std::size_t OW = pad.OutputWidth;
    for (std::size_t oh = 0; oh < OH; ++oh)
    {
        for (std::size_t ow = 0; ow < OW; ++ow)
        {
            const bool inside = (oh >= PT) && (oh < PT + H_IN) &&
                                (ow >= PL) && (ow < PL + W_IN);
            for (std::size_t c = 0; c < C; ++c)
            {
                const int8_t v = out[(oh * OW + ow) * C + c];
                if (inside)
                {
                    const std::size_t ih = oh - PT;
                    const std::size_t iw = ow - PL;
                    BOOST_TEST(v == in[(ih * W_IN + iw) * C + c]);
                }
                else
                {
                    BOOST_TEST(v == static_cast<int8_t>(-7));
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(qconv2d_per_channel_parity_with_per_tensor)
{
    // Single-channel-scale calibration: per-channel should agree with
    // per-tensor when all weight scales are identical.
    constexpr std::size_t H = 4, W = 4, IC = 1, KH = 3, KW = 3, NF = 2;

    const float in_scale = 0.05f;
    const int32_t in_zp = -3;
    const float w_scale = 0.02f;
    const float out_scale = 0.10f;
    const int32_t out_zp = 0;

    int8_t input[H * W * IC];
    for (std::size_t i = 0; i < H * W * IC; ++i)
        input[i] = static_cast<int8_t>((i % 13) - 6);

    int8_t weights[NF * KH * KW * IC];
    for (std::size_t i = 0; i < NF * KH * KW * IC; ++i)
        weights[i] = static_cast<int8_t>((i % 7) - 3);

    int32_t biases[NF] = {0, 0};

    QConv2D<int8_t, int8_t, int32_t, int8_t, H, W, IC, KH, KW, 1, 1, NF> per_tensor;
    per_tensor.weights = weights;
    per_tensor.biases = biases;
    per_tensor.input_zero_point = static_cast<int8_t>(in_zp);
    per_tensor.requantizer = buildRequantizer<int8_t>(in_scale, w_scale, out_scale,
                                                      out_zp, -128, 127);

    QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, IC, KH, KW, 1, 1, NF>
        per_channel;
    Requantizer<int32_t, int8_t> rq[NF];
    for (std::size_t f = 0; f < NF; ++f)
    {
        rq[f] = buildRequantizer<int8_t>(in_scale, w_scale, out_scale,
                                         out_zp, -128, 127);
    }
    per_channel.weights = weights;
    per_channel.biases = biases;
    per_channel.input_zero_point = static_cast<int8_t>(in_zp);
    per_channel.requantizers = rq;

    int8_t out_pt[per_tensor.OutputSize];
    int8_t out_pc[per_channel.OutputSize];
    per_tensor.forward(input, out_pt);
    per_channel.forward(input, out_pc);

    for (std::size_t i = 0; i < per_tensor.OutputSize; ++i)
    {
        BOOST_TEST(out_pt[i] == out_pc[i]);
    }
}

BOOST_AUTO_TEST_CASE(qpointwise_per_channel_parity_with_per_tensor)
{
    constexpr std::size_t H = 3, W = 3, IC = 2, NF = 3;

    const float in_scale = 0.04f;
    const int32_t in_zp = 2;
    const float w_scale = 0.03f;
    const float out_scale = 0.08f;
    const int32_t out_zp = -1;

    int8_t input[H * W * IC];
    for (std::size_t i = 0; i < H * W * IC; ++i)
        input[i] = static_cast<int8_t>((i % 11) - 5);

    int8_t weights[NF * IC];
    for (std::size_t i = 0; i < NF * IC; ++i)
        weights[i] = static_cast<int8_t>((i % 5) - 2);

    int32_t biases[NF] = {0, 0, 0};

    QPointwiseConv2D<int8_t, int8_t, int32_t, int8_t, H, W, IC, NF> per_tensor;
    per_tensor.weights = weights;
    per_tensor.biases = biases;
    per_tensor.input_zero_point = static_cast<int8_t>(in_zp);
    per_tensor.requantizer = buildRequantizer<int8_t>(in_scale, w_scale, out_scale,
                                                      out_zp, -128, 127);

    QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, IC, NF>
        per_channel;
    Requantizer<int32_t, int8_t> rq[NF];
    for (std::size_t f = 0; f < NF; ++f)
    {
        rq[f] = buildRequantizer<int8_t>(in_scale, w_scale, out_scale,
                                         out_zp, -128, 127);
    }
    per_channel.weights = weights;
    per_channel.biases = biases;
    per_channel.input_zero_point = static_cast<int8_t>(in_zp);
    per_channel.requantizers = rq;

    int8_t out_pt[per_tensor.OutputSize];
    int8_t out_pc[per_channel.OutputSize];
    per_tensor.forward(input, out_pt);
    per_channel.forward(input, out_pc);

    for (std::size_t i = 0; i < per_tensor.OutputSize; ++i)
    {
        BOOST_TEST(out_pt[i] == out_pc[i]);
    }
}

BOOST_AUTO_TEST_CASE(qconv2d_per_channel_differs_when_scales_differ)
{
    // Force the two filters onto very different scales. Per-channel
    // result should diverge from the per-tensor approximation that
    // averages them.
    constexpr std::size_t H = 4, W = 4, IC = 1, KH = 3, KW = 3, NF = 2;

    const float in_scale = 0.05f;
    const int32_t in_zp = 0;
    const float w_scale_f0 = 0.01f;
    const float w_scale_f1 = 0.5f;
    const float out_scale = 0.20f;
    const int32_t out_zp = 0;

    int8_t input[H * W * IC];
    for (std::size_t i = 0; i < H * W * IC; ++i) input[i] = static_cast<int8_t>(50);

    int8_t weights[NF * KH * KW * IC];
    for (std::size_t i = 0; i < NF * KH * KW * IC; ++i)
        weights[i] = static_cast<int8_t>(10);
    int32_t biases[NF] = {0, 0};

    QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, H, W, IC, KH, KW, 1, 1, NF>
        per_channel;
    Requantizer<int32_t, int8_t> rq[NF];
    rq[0] = buildRequantizer<int8_t>(in_scale, w_scale_f0, out_scale,
                                     out_zp, -128, 127);
    rq[1] = buildRequantizer<int8_t>(in_scale, w_scale_f1, out_scale,
                                     out_zp, -128, 127);
    per_channel.weights = weights;
    per_channel.biases = biases;
    per_channel.input_zero_point = static_cast<int8_t>(in_zp);
    per_channel.requantizers = rq;

    int8_t out_pc[per_channel.OutputSize];
    per_channel.forward(input, out_pc);

    // Filter 0 (tiny weight scale) should be near zero in int8 land;
    // filter 1 (big weight scale) should saturate or land far from
    // zero. At minimum the two filters must produce different output.
    bool saw_different = false;
    for (std::size_t oh = 0; oh < per_channel.OutputHeight; ++oh)
    {
        for (std::size_t ow = 0; ow < per_channel.OutputWidth; ++ow)
        {
            const std::size_t off = (oh * per_channel.OutputWidth + ow) * NF;
            if (out_pc[off + 0] != out_pc[off + 1]) saw_different = true;
        }
    }
    BOOST_TEST(saw_different);
}

BOOST_AUTO_TEST_CASE(qadd_resnet_residual_shape_smoke)
{
    // Tiny ResNet-block-like fixture: conv -> add -> relu. Verifies
    // the pieces wire together; full per-channel parity is covered
    // above, so this is a smoke test for the data path.
    constexpr std::size_t H = 2, W = 2, C = 2, N = H * W * C;

    const float scale = 0.1f;
    const int32_t zp = 0;

    int8_t branch_x[N], branch_residual[N], sum[N];
    for (std::size_t i = 0; i < N; ++i)
    {
        branch_x[i]        = static_cast<int8_t>(i + 1);
        branch_residual[i] = static_cast<int8_t>(-static_cast<int>(i));
    }

    QAdd<int8_t, int8_t, int8_t, N> add;
    QAddParams p = buildQAddParams(scale, scale, scale);
    add.input_a_zero_point = static_cast<int8_t>(zp);
    add.input_b_zero_point = static_cast<int8_t>(zp);
    add.left_shift = p.left_shift;
    add.input_a_multiplier = p.input_a_multiplier;
    add.input_a_shift = p.input_a_shift;
    add.input_b_multiplier = p.input_b_multiplier;
    add.input_b_shift = p.input_b_shift;
    add.output_requantizer.multiplier = p.output_multiplier;
    add.output_requantizer.shift = p.output_shift;
    add.output_requantizer.zero_point = static_cast<int8_t>(zp);
    add.output_requantizer.qmin = -128;
    add.output_requantizer.qmax = 127;

    add.forward(branch_x, branch_residual, sum);
    qreluBuffer<int8_t>(sum, N, static_cast<int8_t>(zp));

    // All outputs must be >= zero_point after ReLU.
    for (std::size_t i = 0; i < N; ++i)
    {
        BOOST_TEST(sum[i] >= static_cast<int8_t>(zp));
    }
}

#endif // outer TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD (Phase 10 tests)

BOOST_AUTO_TEST_CASE(qinteger_sqrt_basic)
{
    BOOST_TEST(qIntegerSqrt64(0u) == 0u);
    BOOST_TEST(qIntegerSqrt64(1u) == 1u);
    BOOST_TEST(qIntegerSqrt64(4u) == 2u);
    BOOST_TEST(qIntegerSqrt64(16u) == 4u);
    BOOST_TEST(qIntegerSqrt64(100u) == 10u);
    // Non-square inputs floor.
    BOOST_TEST(qIntegerSqrt64(99u) == 9u);
    BOOST_TEST(qIntegerSqrt64(101u) == 10u);
    BOOST_TEST(qIntegerSqrt64(65025u) == 255u);
}

BOOST_AUTO_TEST_CASE(qinv_sqrt_q30_matches_float)
{
    // Spot-check qInvSqrtQ30 against the float reference 1/sqrt(x). The
    // representation is Q1.30 (so the int value = real / 2^30).
    const uint32_t inputs[] = {1u, 4u, 16u, 100u, 10000u, 65025u};
    for (uint32_t x : inputs)
    {
        const uint32_t q = qInvSqrtQ30(x);
        const double real = static_cast<double>(q) /
                            static_cast<double>(1u << 30);
        const double ref = 1.0 / std::sqrt(static_cast<double>(x));
        // 14 bits of sqrt precision through the Newton iteration, so
        // 0.1% relative error is comfortable for everything except x=1
        // where Q1.30 saturates.
        const double rel = std::fabs(real - ref) / ref;
        BOOST_TEST(rel < 0.01,
                   "x=" << x << " q=" << q << " real=" << real
                        << " ref=" << ref);
    }
}

BOOST_AUTO_TEST_CASE(qsoftmax_lut_index_basic)
{
    BOOST_TEST(tinymind::qSoftmaxLUTIndex(0, 0) == 255u);
    BOOST_TEST(tinymind::qSoftmaxLUTIndex(-128, 127) == 0u);
    BOOST_TEST(tinymind::qSoftmaxLUTIndex(127, 127) == 255u);
    // Defensive clamp: positive diff treated as zero.
    BOOST_TEST(tinymind::qSoftmaxLUTIndex(50, 10) == 255u);
}

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

BOOST_AUTO_TEST_CASE(fold_batch_norm_identity_bn_preserves_weights)
{
    // BN with gamma=1, beta=0, mean=0, variance=1, eps=0 is an identity.
    constexpr std::size_t NF = 2;
    constexpr std::size_t KH = 2, KW = 2, IC = 1;
    constexpr std::size_t WPF = KH * KW * IC;

    float weights[NF * WPF] = {
        0.1f, -0.2f, 0.3f, 0.4f,
        0.5f, -0.6f, 0.7f, -0.8f
    };
    float biases[NF] = {0.05f, -0.05f};
    float gamma[NF] = {1.0f, 1.0f};
    float beta[NF]  = {0.0f, 0.0f};
    float mean[NF]  = {0.0f, 0.0f};
    float var[NF]   = {1.0f, 1.0f};

    float fused_w[NF * WPF] = {0};
    float fused_b[NF] = {0};
    foldBatchNorm(weights, biases, gamma, beta, mean, var,
                  0.0f, NF, WPF, fused_w, fused_b);

    for (std::size_t i = 0; i < NF * WPF; ++i)
    {
        BOOST_TEST(std::fabs(fused_w[i] - weights[i]) < 1e-6f);
    }
    for (std::size_t f = 0; f < NF; ++f)
    {
        BOOST_TEST(std::fabs(fused_b[f] - biases[f]) < 1e-6f);
    }
}

BOOST_AUTO_TEST_CASE(fold_batch_norm_parity_vs_unfused_reference)
{
    // Build a non-trivial Conv2D + BN and verify that running fused
    // conv equals running separate conv -> BN within float tolerance.
    constexpr std::size_t NF = 2;
    constexpr std::size_t KH = 2, KW = 2, IC = 1;
    constexpr std::size_t WPF = KH * KW * IC;

    float weights[NF * WPF] = {
        0.5f, 0.25f, -0.125f, 0.75f,
        -0.5f, 1.0f, 0.5f, -0.25f
    };
    float biases[NF] = {0.1f, -0.2f};
    float gamma[NF] = {2.0f, 0.5f};
    float beta[NF]  = {0.1f, -0.1f};
    float mean[NF]  = {0.4f, -0.3f};
    float var[NF]   = {0.81f, 0.49f};
    const float eps = 1e-3f;

    float fused_w[NF * WPF] = {0};
    float fused_b[NF] = {0};
    foldBatchNorm(weights, biases, gamma, beta, mean, var,
                  eps, NF, WPF, fused_w, fused_b);

    // Input "image": a flat patch of size WPF (so the conv output is
    // a scalar per filter).
    float patch[WPF] = {0.3f, -0.1f, 0.7f, 0.2f};
    for (std::size_t f = 0; f < NF; ++f)
    {
        // Unfused path: conv -> BN.
        float conv_out = biases[f];
        for (std::size_t k = 0; k < WPF; ++k)
        {
            conv_out += weights[f * WPF + k] * patch[k];
        }
        const float bn_out = gamma[f] * (conv_out - mean[f]) /
            std::sqrt(var[f] + eps) + beta[f];

        // Fused path.
        float fused_out = fused_b[f];
        for (std::size_t k = 0; k < WPF; ++k)
        {
            fused_out += fused_w[f * WPF + k] * patch[k];
        }

        BOOST_TEST(std::fabs(fused_out - bn_out) < 1e-5f);
    }
}

BOOST_AUTO_TEST_CASE(qbatchnorm2d_parity_vs_float_reference)
{
    // Drive a small NHWC tensor through QBatchNorm2D and compare against
    // the float BN forward formula. Tolerance: max abs error <= 2 on the
    // int8 grid (TFLite reference accepts <= 1 ulp; allow 2 for the int
    // Q1.14 + mult-shift staircase).
    constexpr std::size_t H = 3, W = 3, C = 2;
    constexpr std::size_t N = H * W * C;

    const float input_scale = 0.1f;
    const int32_t input_zp = -10;
    const float output_scale = 0.2f;
    const int32_t output_zp = 5;

    float gamma[C] = {1.4f, 0.6f};
    float beta[C]  = {0.2f, -0.3f};
    float mean[C]  = {0.0f, 0.5f};
    float var[C]   = {0.25f, 0.81f};
    const float eps = 1e-3f;

    QBatchNormChannelParams params[C];
    buildQBatchNormChannelParams(gamma, beta, mean, var, eps,
                                 input_scale, output_scale, C, params);

    int8_t input[N];
    for (std::size_t i = 0; i < N; ++i)
    {
        const int32_t v = static_cast<int32_t>(i) - 12;
        input[i] = static_cast<int8_t>(v);
    }

    QBatchNorm2D<int8_t, int8_t, H, W, C> bn;
    bn.params = params;
    bn.input_zero_point = static_cast<int8_t>(input_zp);
    bn.output_zero_point = static_cast<int8_t>(output_zp);
    bn.qmin = -128;
    bn.qmax = 127;

    int8_t output[N];
    bn.forward(input, output);

    int max_err = 0;
    for (std::size_t h = 0; h < H; ++h)
    {
        for (std::size_t w = 0; w < W; ++w)
        {
            for (std::size_t c = 0; c < C; ++c)
            {
                const std::size_t off = (h * W + w) * C + c;
                const float x_real = input_scale *
                    static_cast<float>(static_cast<int32_t>(input[off]) -
                                       input_zp);
                const float sigma_eff = gamma[c] /
                    std::sqrt(var[c] + eps);
                const float y_real = sigma_eff * (x_real - mean[c]) + beta[c];
                const long expect = std::lround(
                    static_cast<double>(y_real) /
                    static_cast<double>(output_scale)) + output_zp;
                long expect_clamped = expect;
                if (expect_clamped < -128) expect_clamped = -128;
                if (expect_clamped >  127) expect_clamped =  127;
                const int diff = std::abs(static_cast<int>(output[off]) -
                                          static_cast<int>(expect_clamped));
                if (diff > max_err) max_err = diff;
            }
        }
    }
    BOOST_TEST(max_err <= 2);
}

BOOST_AUTO_TEST_CASE(qlayernorm_constant_row_emits_zero_plus_beta)
{
    // A constant row has zero variance; LayerNorm should emit beta (in
    // output domain) plus output_zero_point everywhere. eps_q==1 keeps
    // the inv-sqrt finite; the centered terms are exactly zero so the
    // gamma scaling drops out.
    constexpr std::size_t R = 1, F = 4;

    const float output_scale = 0.05f;
    int16_t gamma_int[F];
    float gamma_f[F] = {1.0f, 1.0f, 1.0f, 1.0f};
    quantizeLayerNormGamma(gamma_f, F, gamma_int);

    int32_t beta_int[F];
    float beta_f[F] = {0.1f, -0.05f, 0.2f, 0.0f};
    quantizeLayerNormBeta(beta_f, F, output_scale, beta_int);

    int32_t out_mult = 0;
    int32_t out_shift = 0;
    buildQLayerNormOutputParams(output_scale, out_mult, out_shift);

    QLayerNorm1D<int8_t, int8_t, R, F> ln;
    ln.gamma = gamma_int;
    ln.beta = beta_int;
    ln.epsilon_q = 1;
    ln.output_multiplier = out_mult;
    ln.output_shift = out_shift;
    ln.output_zero_point = 0;
    ln.qmin = -128;
    ln.qmax = 127;

    int8_t input[F]  = {7, 7, 7, 7};
    int8_t output[F] = {0, 0, 0, 0};
    ln.forward(input, output);

    for (std::size_t i = 0; i < F; ++i)
    {
        BOOST_TEST(static_cast<int>(output[i]) == static_cast<int>(beta_int[i]));
    }
}

BOOST_AUTO_TEST_CASE(qlayernorm_parity_vs_float_reference)
{
    // Drive a single-row LayerNorm and check that the int8 output is
    // within a few ulps of the rounded float reference. eps quantized
    // to var_q domain is small (rounds to 1 here); both gamma and beta
    // are non-trivial.
    constexpr std::size_t R = 1, F = 8;

    const float input_scale = 0.1f;
    const float output_scale = 0.05f;
    const int32_t output_zp = 0;

    float gamma_f[F] = {1.0f, 1.0f, 1.0f, 1.0f,
                        1.0f, 1.0f, 1.0f, 1.0f};
    float beta_f[F]  = {0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f};

    int16_t gamma_int[F];
    int32_t beta_int[F];
    quantizeLayerNormGamma(gamma_f, F, gamma_int);
    quantizeLayerNormBeta(beta_f, F, output_scale, beta_int);

    int32_t out_mult = 0;
    int32_t out_shift = 0;
    buildQLayerNormOutputParams(output_scale, out_mult, out_shift);
    const int32_t eps_q = quantizeLayerNormEpsilon(1e-5f, input_scale);

    QLayerNorm1D<int8_t, int8_t, R, F> ln;
    ln.gamma = gamma_int;
    ln.beta = beta_int;
    ln.epsilon_q = eps_q;
    ln.output_multiplier = out_mult;
    ln.output_shift = out_shift;
    ln.output_zero_point = static_cast<int8_t>(output_zp);
    ln.qmin = -128;
    ln.qmax = 127;

    int8_t input[F]  = {-30, -10, 0, 10, 20, 30, 40, 50};
    int8_t output[F] = {0};
    ln.forward(input, output);

    // Float reference: mean, var, normalize, gamma, beta, requantize.
    float reals[F];
    for (std::size_t i = 0; i < F; ++i)
    {
        reals[i] = input_scale * static_cast<float>(input[i]);
    }
    float sum = 0;
    for (std::size_t i = 0; i < F; ++i) sum += reals[i];
    const float m = sum / static_cast<float>(F);
    float ss = 0;
    for (std::size_t i = 0; i < F; ++i)
    {
        const float d = reals[i] - m;
        ss += d * d;
    }
    const float vr = ss / static_cast<float>(F);
    const float invs = 1.0f / std::sqrt(vr + 1e-5f);

    int max_err = 0;
    for (std::size_t i = 0; i < F; ++i)
    {
        const float y_real = gamma_f[i] * (reals[i] - m) * invs + beta_f[i];
        long expect = std::lround(
            static_cast<double>(y_real) /
            static_cast<double>(output_scale)) + output_zp;
        if (expect < -128) expect = -128;
        if (expect >  127) expect =  127;
        const int diff = std::abs(static_cast<int>(output[i]) -
                                  static_cast<int>(expect));
        if (diff > max_err) max_err = diff;
    }
    // Phase 11 tolerance is "max abs error <= 1 on int8 grid" in spirit;
    // the integer rsqrt + per-element rescale stack can drift one ulp
    // further on small-magnitude features, so allow 3.
    BOOST_TEST(max_err <= 3);
}

BOOST_AUTO_TEST_CASE(qsoftmax_parity_vs_float_reference)
{
    // Build a calibrated exp LUT and compare integer softmax to the
    // float reference on a small row. Output uses the TFLite convention
    // (scale = 1/256, zero_point = -128).
    constexpr std::size_t R = 1, F = 6;
    const float input_scale = 0.1f;

    int32_t exp_lut[kQSoftmaxExpLUTSize];
    buildQSoftmaxExpLUT(input_scale, exp_lut);

    QSoftmax1D<int8_t, int8_t, R, F> sm;
    sm.exp_lut = exp_lut;
    sm.output_zero_point = -128;
    sm.qmin = -128;
    sm.qmax = 127;

    int8_t input[F]  = {-30, -10, 0, 5, 20, 40};
    int8_t output[F] = {0};
    sm.forward(input, output);

    // Float reference.
    double max_real = input_scale * static_cast<double>(input[0]);
    for (std::size_t i = 1; i < F; ++i)
    {
        const double v = input_scale * static_cast<double>(input[i]);
        if (v > max_real) max_real = v;
    }
    double sum = 0;
    double exps[F];
    for (std::size_t i = 0; i < F; ++i)
    {
        const double v = input_scale * static_cast<double>(input[i]);
        exps[i] = std::exp(v - max_real);
        sum += exps[i];
    }

    int max_err = 0;
    double prob_sum = 0;
    for (std::size_t i = 0; i < F; ++i)
    {
        const double prob = exps[i] / sum;
        prob_sum += prob;
        long expect = std::lround(prob * 256.0) - 128;
        if (expect < -128) expect = -128;
        if (expect >  127) expect =  127;
        const int diff = std::abs(static_cast<int>(output[i]) -
                                  static_cast<int>(expect));
        if (diff > max_err) max_err = diff;
    }
    BOOST_TEST(max_err <= 2);
    BOOST_TEST(std::fabs(prob_sum - 1.0) < 1e-9);
}

BOOST_AUTO_TEST_CASE(qsoftmax_dominant_class_saturates_high)
{
    // One element dwarfs the rest -> its probability rounds to 1.0;
    // q_out = 256 - 128 = 128, saturated to 127.
    constexpr std::size_t R = 1, F = 4;
    const float input_scale = 0.5f;

    int32_t exp_lut[kQSoftmaxExpLUTSize];
    buildQSoftmaxExpLUT(input_scale, exp_lut);

    QSoftmax1D<int8_t, int8_t, R, F> sm;
    sm.exp_lut = exp_lut;
    sm.output_zero_point = -128;
    sm.qmin = -128;
    sm.qmax = 127;

    int8_t input[F]  = {-127, -127, -127, 127};
    int8_t output[F] = {0};
    sm.forward(input, output);

    BOOST_TEST(static_cast<int>(output[3]) == 127);
    for (std::size_t i = 0; i < 3; ++i)
    {
        BOOST_TEST(static_cast<int>(output[i]) == -128);
    }
}

BOOST_AUTO_TEST_CASE(qsoftmax_zero_lut_emits_clamped_zero_point)
{
    // An all-zero exp LUT (degenerate / mis-calibrated input_scale) makes the
    // row sum 0; the forward must take the divide-by-zero guard and emit the
    // clamped output zero_point rather than dividing.
    constexpr std::size_t R = 1, F = 4;

    int32_t exp_lut[kQSoftmaxExpLUTSize] = {0};

    QSoftmax1D<int8_t, int8_t, R, F> sm;
    sm.exp_lut = exp_lut;
    sm.output_zero_point = -128;
    sm.qmin = -128;
    sm.qmax = 127;

    int8_t input[F]  = {10, -20, 30, 0};
    int8_t output[F] = {1, 1, 1, 1};
    sm.forward(input, output);

    for (std::size_t i = 0; i < F; ++i)
    {
        BOOST_TEST(static_cast<int>(output[i]) == -128);
    }
}

namespace {

// ----- Phase 12 recurrent quantization test helpers ----------------------

constexpr std::size_t kLstmInputs  = 3;
constexpr std::size_t kLstmHidden  = 4;

inline float sigmoidf(float x)  { return 1.0f / (1.0f + std::exp(-x)); }
inline float tanhf_(float x)    { return std::tanh(x); }

void floatLstmReference(const float* x,
                        float* h_state,
                        float* c_state,
                        const float* w_in,
                        const float* w_rec,
                        const float* bias)
{
    const std::size_t I = kLstmInputs;
    const std::size_t H = kLstmHidden;
    float pre[4 * H];
    for (std::size_t g = 0; g < 4; ++g)
    {
        for (std::size_t hi = 0; hi < H; ++hi)
        {
            float acc = bias[g * H + hi];
            for (std::size_t j = 0; j < I; ++j)
            {
                acc += w_in[g * H * I + hi * I + j] * x[j];
            }
            for (std::size_t k = 0; k < H; ++k)
            {
                acc += w_rec[g * H * H + hi * H + k] * h_state[k];
            }
            pre[g * H + hi] = acc;
        }
    }
    for (std::size_t hi = 0; hi < H; ++hi)
    {
        const float i_t = sigmoidf(pre[0 * H + hi]);
        const float f_t = sigmoidf(pre[1 * H + hi]);
        const float g_t = tanhf_  (pre[2 * H + hi]);
        const float o_t = sigmoidf(pre[3 * H + hi]);
        const float c_new = f_t * c_state[hi] + i_t * g_t;
        c_state[hi] = c_new;
        h_state[hi] = o_t * tanhf_(c_new);
    }
}

float absmax(const float* p, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        const float a = std::fabs(p[i]);
        if (a > m) m = a;
    }
    return m;
}

void quantizeSymmetricWeights(const float* src, int8_t* dst, std::size_t n,
                              float scale)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        const long q = std::lround(
            static_cast<double>(src[i]) / static_cast<double>(scale));
        long c = q;
        if (c < -127) c = -127;
        if (c >  127) c =  127;
        dst[i] = static_cast<int8_t>(c);
    }
}

// Deterministic small-magnitude weight/bias filler. Values stay inside the
// linear region of sigmoid/tanh so the LUT calibration error stays bounded.
void fillLstmWeights(float* w_in, float* w_rec, float* bias)
{
    for (std::size_t g = 0; g < 4; ++g)
    {
        for (std::size_t hi = 0; hi < kLstmHidden; ++hi)
        {
            for (std::size_t j = 0; j < kLstmInputs; ++j)
            {
                const std::size_t idx = g * kLstmHidden * kLstmInputs +
                                        hi * kLstmInputs + j;
                const int32_t mod = static_cast<int32_t>(idx % 7);
                w_in[idx] = 0.10f * static_cast<float>(mod - 3);
            }
            for (std::size_t k = 0; k < kLstmHidden; ++k)
            {
                const std::size_t idx = g * kLstmHidden * kLstmHidden +
                                        hi * kLstmHidden + k;
                const int32_t mod = static_cast<int32_t>(idx % 5);
                w_rec[idx] = 0.08f * static_cast<float>(mod - 2);
            }
            const std::size_t bidx = g * kLstmHidden + hi;
            const int32_t mod = static_cast<int32_t>(bidx % 3);
            bias[bidx] = 0.05f * static_cast<float>(mod - 1);
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(qlstm_single_step_parity_with_float_reference)
{
    constexpr std::size_t I = kLstmInputs;
    constexpr std::size_t H = kLstmHidden;

    float w_in [4 * H * I];
    float w_rec[4 * H * H];
    float bias [4 * H];
    fillLstmWeights(w_in, w_rec, bias);

    // Run float reference for the comparison ground truth.
    float x_ref[I]    = { 0.5f, -0.3f, 0.7f };
    float h_ref[H]    = { 0.0f, 0.0f, 0.0f, 0.0f };
    float c_ref[H]    = { 0.0f, 0.0f, 0.0f, 0.0f };
    floatLstmReference(x_ref, h_ref, c_ref, w_in, w_rec, bias);

    // Calibrate. Wide LUT input range to cover the linear+saturation regions
    // of sigmoid/tanh; per-gate symmetric weight scales.
    const float x_scale   = 1.0f / 127.0f;
    const float h_scale   = 1.0f / 127.0f;
    const float c_scale   = 4.0f / 127.0f;
    const float lut_scale = 8.0f / 127.0f;

    QLSTMScales sc;
    sc.input_scale     = x_scale;
    sc.hidden_scale    = h_scale;
    sc.cell_scale      = c_scale;
    sc.lut_input_scale = lut_scale;
    for (std::size_t g = 0; g < 4; ++g)
    {
        sc.w_input_scale[g] =
            absmax(w_in  + g * H * I, H * I) / 127.0f;
        sc.w_recurrent_scale[g] =
            absmax(w_rec + g * H * H, H * H) / 127.0f;
    }

    QLSTMParams p;
    buildQLSTMParams(sc, p);

    int32_t biases_q[4 * H];
    quantizeQLSTMBiases(bias, H, lut_scale, biases_q);

    int8_t w_in_q [4 * H * I];
    int8_t w_rec_q[4 * H * H];
    for (std::size_t g = 0; g < 4; ++g)
    {
        quantizeSymmetricWeights(w_in  + g * H * I,
                                 w_in_q + g * H * I, H * I,
                                 sc.w_input_scale[g]);
        quantizeSymmetricWeights(w_rec + g * H * H,
                                 w_rec_q + g * H * H, H * H,
                                 sc.w_recurrent_scale[g]);
    }

    int8_t sigmoid_lut[256], tanh_lut[256], tanh_cell_lut[256];
    buildQSigmoidLUT(lut_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_lut);
    buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_cell_lut);

    tinymind::QLSTMCell<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t, I, H>
        cell;
    cell.w_input     = w_in_q;
    cell.w_recurrent = w_rec_q;
    cell.biases      = biases_q;
    cell.input_zero_point  = 0;
    cell.hidden_zero_point = 0;
    cell.cell_zero_point   = 0;
    for (std::size_t g = 0; g < 4; ++g)
    {
        cell.input_to_lut_multiplier   [g] = p.input_to_lut_multiplier[g];
        cell.input_to_lut_shift        [g] = p.input_to_lut_shift     [g];
        cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
        cell.recurrent_to_lut_shift    [g] = p.recurrent_to_lut_shift     [g];
    }
    cell.sigmoid_lut             = sigmoid_lut;
    cell.tanh_lut                = tanh_lut;
    cell.tanh_cell_lut           = tanh_cell_lut;
    cell.f_times_c_multiplier    = p.f_times_c_multiplier;
    cell.f_times_c_shift         = p.f_times_c_shift;
    cell.i_times_g_multiplier    = p.i_times_g_multiplier;
    cell.i_times_g_shift         = p.i_times_g_shift;
    cell.cell_qmin               = -127;
    cell.cell_qmax               =  127;
    cell.cell_to_tanh_multiplier = p.cell_to_tanh_multiplier;
    cell.cell_to_tanh_shift      = p.cell_to_tanh_shift;
    cell.output_requantizer.multiplier = p.output_multiplier;
    cell.output_requantizer.shift      = p.output_shift;
    cell.output_requantizer.zero_point = 0;
    cell.output_requantizer.qmin       = -128;
    cell.output_requantizer.qmax       =  127;

    int8_t x_q[I];
    quantizeBuffer<int8_t>(x_ref, x_q, I, x_scale, 0, -128, 127);
    int8_t h_q[H] = {0, 0, 0, 0};
    int8_t c_q[H] = {0, 0, 0, 0};
    cell.forward(x_q, h_q, c_q);

    float max_h_err = 0.0f;
    float max_c_err = 0.0f;
    for (std::size_t hi = 0; hi < H; ++hi)
    {
        const float h_back = dequantize<int8_t>(h_q[hi], h_scale, 0);
        const float c_back = dequantize<int8_t>(c_q[hi], c_scale, 0);
        const float e_h = std::fabs(h_back - h_ref[hi]);
        const float e_c = std::fabs(c_back - c_ref[hi]);
        if (e_h > max_h_err) max_h_err = e_h;
        if (e_c > max_c_err) max_c_err = e_c;
    }

    // Tolerances reflect compounded LUT + per-gate requantization error
    // for an int8 single-step cell. Empirically a few percent of full
    // hidden / cell dynamic range is the noise floor.
    BOOST_TEST(max_h_err < 0.05f);
    BOOST_TEST(max_c_err < 0.15f);
}

BOOST_AUTO_TEST_CASE(qlstm_int16_cell_long_sequence_stays_bounded)
{
    // Drive the int16 cell-state variant for many timesteps and confirm
    // it does not saturate. The int8 variant would clip the cell state
    // when the magnitude approaches the cell quantization grid; the int16
    // variant has 256x more headroom for the carry-state accumulator.
    constexpr std::size_t I = kLstmInputs;
    constexpr std::size_t H = kLstmHidden;
    constexpr std::size_t T = 256;

    float w_in [4 * H * I];
    float w_rec[4 * H * H];
    float bias [4 * H];
    fillLstmWeights(w_in, w_rec, bias);

    const float x_scale   = 1.0f / 127.0f;
    const float h_scale   = 1.0f / 127.0f;
    const float c_scale   = 4.0f / 32767.0f;
    const float lut_scale = 8.0f / 127.0f;

    QLSTMScales sc;
    sc.input_scale     = x_scale;
    sc.hidden_scale    = h_scale;
    sc.cell_scale      = c_scale;
    sc.lut_input_scale = lut_scale;
    for (std::size_t g = 0; g < 4; ++g)
    {
        sc.w_input_scale[g] =
            absmax(w_in  + g * H * I, H * I) / 127.0f;
        sc.w_recurrent_scale[g] =
            absmax(w_rec + g * H * H, H * H) / 127.0f;
    }

    QLSTMParams p;
    buildQLSTMParams(sc, p);

    int32_t biases_q[4 * H];
    quantizeQLSTMBiases(bias, H, lut_scale, biases_q);

    int8_t w_in_q [4 * H * I];
    int8_t w_rec_q[4 * H * H];
    for (std::size_t g = 0; g < 4; ++g)
    {
        quantizeSymmetricWeights(w_in  + g * H * I,
                                 w_in_q + g * H * I, H * I,
                                 sc.w_input_scale[g]);
        quantizeSymmetricWeights(w_rec + g * H * H,
                                 w_rec_q + g * H * H, H * H,
                                 sc.w_recurrent_scale[g]);
    }

    int8_t sigmoid_lut[256], tanh_lut[256], tanh_cell_lut[256];
    buildQSigmoidLUT(lut_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_lut);
    buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_cell_lut);

    tinymind::QLSTMCell<int8_t, int8_t, int32_t, int8_t, int8_t, int16_t, I, H>
        cell;
    cell.w_input     = w_in_q;
    cell.w_recurrent = w_rec_q;
    cell.biases      = biases_q;
    cell.input_zero_point  = 0;
    cell.hidden_zero_point = 0;
    cell.cell_zero_point   = 0;
    for (std::size_t g = 0; g < 4; ++g)
    {
        cell.input_to_lut_multiplier   [g] = p.input_to_lut_multiplier[g];
        cell.input_to_lut_shift        [g] = p.input_to_lut_shift     [g];
        cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
        cell.recurrent_to_lut_shift    [g] = p.recurrent_to_lut_shift     [g];
    }
    cell.sigmoid_lut             = sigmoid_lut;
    cell.tanh_lut                = tanh_lut;
    cell.tanh_cell_lut           = tanh_cell_lut;
    cell.f_times_c_multiplier    = p.f_times_c_multiplier;
    cell.f_times_c_shift         = p.f_times_c_shift;
    cell.i_times_g_multiplier    = p.i_times_g_multiplier;
    cell.i_times_g_shift         = p.i_times_g_shift;
    cell.cell_qmin               = -32767;
    cell.cell_qmax               =  32767;
    cell.cell_to_tanh_multiplier = p.cell_to_tanh_multiplier;
    cell.cell_to_tanh_shift      = p.cell_to_tanh_shift;
    cell.output_requantizer.multiplier = p.output_multiplier;
    cell.output_requantizer.shift      = p.output_shift;
    cell.output_requantizer.zero_point = 0;
    cell.output_requantizer.qmin       = -128;
    cell.output_requantizer.qmax       =  127;

    int8_t  x_q[I];
    int8_t  h_q[H] = {0, 0, 0, 0};
    int16_t c_q[H] = {0, 0, 0, 0};
    float   h_ref_state[H] = {0, 0, 0, 0};
    float   c_ref_state[H] = {0, 0, 0, 0};
    float max_c_err = 0.0f;

    for (std::size_t t = 0; t < T; ++t)
    {
        const int32_t m3  = static_cast<int32_t>(t % 3);
        const int32_t m5  = static_cast<int32_t>(t % 5);
        const int32_t m23 = static_cast<int32_t>((t / 2) % 3);
        const float x_ref[I] = {
            0.5f  * static_cast<float>(m3  - 1),
            -0.4f * static_cast<float>(m5  - 2),
            0.6f  * static_cast<float>(m23 - 1)
        };
        quantizeBuffer<int8_t>(x_ref, x_q, I, x_scale, 0, -128, 127);

        floatLstmReference(x_ref, h_ref_state, c_ref_state,
                           w_in, w_rec, bias);
        cell.forward(x_q, h_q, c_q);

        for (std::size_t hi = 0; hi < H; ++hi)
        {
            const float c_back = dequantize<int16_t>(c_q[hi], c_scale, 0);
            const float e_c = std::fabs(c_back - c_ref_state[hi]);
            if (e_c > max_c_err) max_c_err = e_c;
        }
    }

    // int16 cell keeps the carry-state drift small over 256 steps; loose
    // bound to absorb LUT quantization error compounded across time.
    BOOST_TEST(max_c_err < 0.5f);
    // Nothing pinned to the cell saturation rail.
    for (std::size_t hi = 0; hi < H; ++hi)
    {
        BOOST_TEST(c_q[hi] >  -32000);
        BOOST_TEST(c_q[hi] <   32000);
    }
}

namespace {

constexpr std::size_t kGruInputs = 3;
constexpr std::size_t kGruHidden = 4;

void floatGruReference(const float* x,
                       float* h_state,
                       const float* w_in,
                       const float* w_rec,
                       const float* bias)
{
    const std::size_t I = kGruInputs;
    const std::size_t H = kGruHidden;
    auto sig = [](float v){ return 1.0f / (1.0f + std::exp(-v)); };
    auto tnh = [](float v){ return std::tanh(v); };

    float r[H], z[H];
    float rh[H];
    float n[H];

    for (std::size_t gate = 0; gate < 2; ++gate)
    {
        const std::size_t g = gate; // r=0, z=1
        for (std::size_t hi = 0; hi < H; ++hi)
        {
            float acc = bias[g * H + hi];
            for (std::size_t j = 0; j < I; ++j)
                acc += w_in[g * H * I + hi * I + j] * x[j];
            for (std::size_t k = 0; k < H; ++k)
                acc += w_rec[g * H * H + hi * H + k] * h_state[k];
            const float s = sig(acc);
            if (gate == 0) r[hi] = s;
            else           z[hi] = s;
        }
    }

    for (std::size_t k = 0; k < H; ++k)
    {
        rh[k] = r[k] * h_state[k];
    }

    for (std::size_t hi = 0; hi < H; ++hi)
    {
        float acc = bias[2 * H + hi];
        for (std::size_t j = 0; j < I; ++j)
            acc += w_in[2 * H * I + hi * I + j] * x[j];
        for (std::size_t k = 0; k < H; ++k)
            acc += w_rec[2 * H * H + hi * H + k] * rh[k];
        n[hi] = tnh(acc);
    }

    for (std::size_t hi = 0; hi < H; ++hi)
    {
        h_state[hi] = (1.0f - z[hi]) * n[hi] + z[hi] * h_state[hi];
    }
}

void fillGruWeights(float* w_in, float* w_rec, float* bias)
{
    for (std::size_t g = 0; g < 3; ++g)
    {
        for (std::size_t hi = 0; hi < kGruHidden; ++hi)
        {
            for (std::size_t j = 0; j < kGruInputs; ++j)
            {
                const std::size_t idx = g * kGruHidden * kGruInputs +
                                        hi * kGruInputs + j;
                const int32_t mod = static_cast<int32_t>(idx % 7);
                w_in[idx] = 0.09f * static_cast<float>(mod - 3);
            }
            for (std::size_t k = 0; k < kGruHidden; ++k)
            {
                const std::size_t idx = g * kGruHidden * kGruHidden +
                                        hi * kGruHidden + k;
                const int32_t mod = static_cast<int32_t>(idx % 5);
                w_rec[idx] = 0.07f * static_cast<float>(mod - 2);
            }
            const std::size_t bidx = g * kGruHidden + hi;
            const int32_t mod = static_cast<int32_t>(bidx % 3);
            bias[bidx] = 0.04f * static_cast<float>(mod - 1);
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(qgru_single_step_parity_with_float_reference)
{
    constexpr std::size_t I = kGruInputs;
    constexpr std::size_t H = kGruHidden;

    float w_in [3 * H * I];
    float w_rec[3 * H * H];
    float bias [3 * H];
    fillGruWeights(w_in, w_rec, bias);

    float x_ref[I] = { 0.4f, -0.2f, 0.6f };
    float h_ref[H] = { 0.0f, 0.0f, 0.0f, 0.0f };
    floatGruReference(x_ref, h_ref, w_in, w_rec, bias);

    const float x_scale   = 1.0f / 127.0f;
    const float h_scale   = 1.0f / 127.0f;
    const float lut_scale = 8.0f / 127.0f;

    QGRUScales sc;
    sc.input_scale     = x_scale;
    sc.hidden_scale    = h_scale;
    sc.lut_input_scale = lut_scale;
    for (std::size_t g = 0; g < 3; ++g)
    {
        sc.w_input_scale[g] =
            absmax(w_in  + g * H * I, H * I) / 127.0f;
        sc.w_recurrent_scale[g] =
            absmax(w_rec + g * H * H, H * H) / 127.0f;
    }

    QGRUParams p;
    buildQGRUParams(sc, p);

    int32_t biases_q[3 * H];
    quantizeQGRUBiases(bias, H, lut_scale, biases_q);

    int8_t w_in_q [3 * H * I];
    int8_t w_rec_q[3 * H * H];
    for (std::size_t g = 0; g < 3; ++g)
    {
        quantizeSymmetricWeights(w_in  + g * H * I,
                                 w_in_q + g * H * I, H * I,
                                 sc.w_input_scale[g]);
        quantizeSymmetricWeights(w_rec + g * H * H,
                                 w_rec_q + g * H * H, H * H,
                                 sc.w_recurrent_scale[g]);
    }

    int8_t sigmoid_lut[256], tanh_lut[256];
    buildQSigmoidLUT(lut_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_lut);

    tinymind::QGRUCell<int8_t, int8_t, int32_t, int8_t, int8_t, I, H> cell;
    cell.w_input     = w_in_q;
    cell.w_recurrent = w_rec_q;
    cell.biases      = biases_q;
    cell.input_zero_point  = 0;
    cell.hidden_zero_point = 0;
    for (std::size_t g = 0; g < 3; ++g)
    {
        cell.input_to_lut_multiplier   [g] = p.input_to_lut_multiplier[g];
        cell.input_to_lut_shift        [g] = p.input_to_lut_shift     [g];
        cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
        cell.recurrent_to_lut_shift    [g] = p.recurrent_to_lut_shift     [g];
    }
    cell.sigmoid_lut                    = sigmoid_lut;
    cell.tanh_lut                       = tanh_lut;
    cell.r_times_h_multiplier           = p.r_times_h_multiplier;
    cell.r_times_h_shift                = p.r_times_h_shift;
    cell.one_minus_z_times_n_multiplier = p.one_minus_z_times_n_multiplier;
    cell.one_minus_z_times_n_shift      = p.one_minus_z_times_n_shift;
    cell.z_times_h_multiplier           = p.z_times_h_multiplier;
    cell.z_times_h_shift                = p.z_times_h_shift;
    cell.output_qmin                    = -127;
    cell.output_qmax                    =  127;

    int8_t x_q[I];
    quantizeBuffer<int8_t>(x_ref, x_q, I, x_scale, 0, -128, 127);
    int8_t h_q[H] = {0, 0, 0, 0};
    cell.forward(x_q, h_q);

    float max_h_err = 0.0f;
    for (std::size_t hi = 0; hi < H; ++hi)
    {
        const float h_back = dequantize<int8_t>(h_q[hi], h_scale, 0);
        const float e = std::fabs(h_back - h_ref[hi]);
        if (e > max_h_err) max_h_err = e;
    }
    BOOST_TEST(max_h_err < 0.05f);
}

// ----------------------------------------------------------------------------
// QCfC (closed-form continuous-time) single-step parity vs a float reference.
// ----------------------------------------------------------------------------
namespace {

constexpr std::size_t kCfcInputs   = 2;
constexpr std::size_t kCfcHidden   = 3;
constexpr std::size_t kCfcBackbone = 4;

void fillCfcWeights(float* w_bin, float* w_bh,
                    float* w_ff1, float* w_ff2, float* w_ta, float* w_tb,
                    float* b_bb, float* b_ff1, float* b_ff2,
                    float* b_a, float* b_b)
{
    // Deterministic small weights so tanh stays in its near-linear range and
    // the int8 grid resolves the activations cleanly.
    unsigned s = 24601u;
    auto nxt = [&s]() {
        s = s * 1103515245u + 12345u;
        return 0.30f * ((static_cast<float>((s >> 16) & 0x7fff) / 32767.0f) - 0.5f);
    };
    for (std::size_t i = 0; i < kCfcBackbone * kCfcInputs; ++i) w_bin[i] = nxt();
    for (std::size_t i = 0; i < kCfcBackbone * kCfcHidden; ++i) w_bh[i]  = nxt();
    for (std::size_t i = 0; i < kCfcHidden * kCfcBackbone; ++i) { w_ff1[i]=nxt(); w_ff2[i]=nxt(); w_ta[i]=nxt(); w_tb[i]=nxt(); }
    for (std::size_t i = 0; i < kCfcBackbone; ++i) b_bb[i] = nxt();
    for (std::size_t i = 0; i < kCfcHidden; ++i) { b_ff1[i]=nxt(); b_ff2[i]=nxt(); b_a[i]=nxt(); b_b[i]=nxt(); }
}

void floatCfcReference(const float* x, float* h_state, float ts,
                       const float* w_bin, const float* w_bh,
                       const float* w_ff1, const float* w_ff2,
                       const float* w_ta, const float* w_tb,
                       const float* b_bb, const float* b_ff1, const float* b_ff2,
                       const float* b_a, const float* b_b)
{
    const std::size_t I = kCfcInputs, H = kCfcHidden, BB = kCfcBackbone;
    auto sig = [](float v){ return 1.0f / (1.0f + std::exp(-v)); };

    float x1[BB];
    for (std::size_t u = 0; u < BB; ++u)
    {
        float z = b_bb[u];
        for (std::size_t j = 0; j < I; ++j) z += w_bin[u * I + j] * x[j];
        for (std::size_t k = 0; k < H; ++k) z += w_bh[u * H + k] * h_state[k];
        x1[u] = std::tanh(z);
    }
    float out[H];
    for (std::size_t i = 0; i < H; ++i)
    {
        float a1 = b_ff1[i], a2 = b_ff2[i], aA = b_a[i], aB = b_b[i];
        for (std::size_t u = 0; u < BB; ++u)
        {
            a1 += w_ff1[i * BB + u] * x1[u];
            a2 += w_ff2[i * BB + u] * x1[u];
            aA += w_ta [i * BB + u] * x1[u];
            aB += w_tb [i * BB + u] * x1[u];
        }
        const float ff1 = std::tanh(a1);
        const float ff2 = std::tanh(a2);
        const float t   = sig(aA * ts + aB);
        out[i] = (1.0f - t) * ff1 + t * ff2;
    }
    for (std::size_t i = 0; i < H; ++i) h_state[i] = out[i];
}

} // namespace

BOOST_AUTO_TEST_CASE(qcfc_single_step_parity_with_float_reference)
{
    constexpr std::size_t I = kCfcInputs, H = kCfcHidden, BB = kCfcBackbone;
    const float ts = 1.5f;

    float w_bin[BB * I], w_bh[BB * H];
    float w_ff1[H * BB], w_ff2[H * BB], w_ta[H * BB], w_tb[H * BB];
    float b_bb[BB], b_ff1[H], b_ff2[H], b_a[H], b_b[H];
    fillCfcWeights(w_bin, w_bh, w_ff1, w_ff2, w_ta, w_tb, b_bb, b_ff1, b_ff2, b_a, b_b);

    float x_ref[I] = { 0.5f, -0.3f };
    float h_ref[H] = { 0.0f, 0.0f, 0.0f };
    floatCfcReference(x_ref, h_ref, ts, w_bin, w_bh, w_ff1, w_ff2, w_ta, w_tb,
                      b_bb, b_ff1, b_ff2, b_a, b_b);

    const float x_scale = 1.0f / 127.0f, h_scale = 1.0f / 127.0f;
    const float lut_scale = 8.0f / 127.0f;

    QCfCScales sc;
    sc.input_scale  = x_scale;
    sc.hidden_scale = h_scale;
    sc.lut_input_scale = lut_scale;
    sc.w_backbone_input_scale  = absmax(w_bin, BB * I) / 127.0f;
    sc.w_backbone_hidden_scale = absmax(w_bh,  BB * H) / 127.0f;
    sc.w_ff1_scale    = absmax(w_ff1, H * BB) / 127.0f;
    sc.w_ff2_scale    = absmax(w_ff2, H * BB) / 127.0f;
    sc.w_time_a_scale = absmax(w_ta,  H * BB) / 127.0f;
    sc.w_time_b_scale = absmax(w_tb,  H * BB) / 127.0f;
    sc.ts = static_cast<double>(ts);

    QCfCParams p;
    buildQCfCParams(sc, p);

    int32_t b_bb_q[BB], b_ff1_q[H], b_ff2_q[H], b_time_q[H];
    quantizeQCfCBias(b_bb,  BB, lut_scale, b_bb_q);
    quantizeQCfCBias(b_ff1, H,  lut_scale, b_ff1_q);
    quantizeQCfCBias(b_ff2, H,  lut_scale, b_ff2_q);
    quantizeQCfCTimeBias(b_a, b_b, H, sc.ts, lut_scale, b_time_q);

    int8_t w_bin_q[BB * I], w_bh_q[BB * H];
    int8_t w_ff1_q[H * BB], w_ff2_q[H * BB], w_ta_q[H * BB], w_tb_q[H * BB];
    quantizeSymmetricWeights(w_bin, w_bin_q, BB * I, sc.w_backbone_input_scale);
    quantizeSymmetricWeights(w_bh,  w_bh_q,  BB * H, sc.w_backbone_hidden_scale);
    quantizeSymmetricWeights(w_ff1, w_ff1_q, H * BB, sc.w_ff1_scale);
    quantizeSymmetricWeights(w_ff2, w_ff2_q, H * BB, sc.w_ff2_scale);
    quantizeSymmetricWeights(w_ta,  w_ta_q,  H * BB, sc.w_time_a_scale);
    quantizeSymmetricWeights(w_tb,  w_tb_q,  H * BB, sc.w_time_b_scale);

    int8_t sigmoid_lut[256], tanh_lut[256];
    buildQSigmoidLUT(lut_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
    buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_lut);

    tinymind::QCfCCell<int8_t, int8_t, int32_t, int8_t, I, H, BB> cell;
    cell.w_backbone_input  = w_bin_q;
    cell.w_backbone_hidden = w_bh_q;
    cell.b_backbone        = b_bb_q;
    cell.w_ff1 = w_ff1_q; cell.w_ff2 = w_ff2_q;
    cell.w_time_a = w_ta_q; cell.w_time_b = w_tb_q;
    cell.b_ff1 = b_ff1_q; cell.b_ff2 = b_ff2_q; cell.b_time = b_time_q;
    cell.input_zero_point = 0; cell.hidden_zero_point = 0;
    cell.backbone_input_multiplier  = p.backbone_input_multiplier;
    cell.backbone_input_shift       = p.backbone_input_shift;
    cell.backbone_hidden_multiplier = p.backbone_hidden_multiplier;
    cell.backbone_hidden_shift      = p.backbone_hidden_shift;
    cell.ff1_multiplier = p.ff1_multiplier; cell.ff1_shift = p.ff1_shift;
    cell.ff2_multiplier = p.ff2_multiplier; cell.ff2_shift = p.ff2_shift;
    cell.time_a_multiplier = p.time_a_multiplier; cell.time_a_shift = p.time_a_shift;
    cell.time_b_multiplier = p.time_b_multiplier; cell.time_b_shift = p.time_b_shift;
    cell.sigmoid_lut = sigmoid_lut; cell.tanh_lut = tanh_lut;
    cell.one_minus_t_times_ff1_multiplier = p.one_minus_t_times_ff1_multiplier;
    cell.one_minus_t_times_ff1_shift      = p.one_minus_t_times_ff1_shift;
    cell.t_times_ff2_multiplier = p.t_times_ff2_multiplier;
    cell.t_times_ff2_shift      = p.t_times_ff2_shift;
    cell.output_qmin = -127; cell.output_qmax = 127;

    int8_t x_q[I];
    quantizeBuffer<int8_t>(x_ref, x_q, I, x_scale, 0, -128, 127);
    int8_t h_q[H] = {0, 0, 0};
    cell.forward(x_q, h_q);

    float max_h_err = 0.0f;
    for (std::size_t hi = 0; hi < H; ++hi)
    {
        const float h_back = dequantize<int8_t>(h_q[hi], h_scale, 0);
        const float e = std::fabs(h_back - h_ref[hi]);
        if (e > max_h_err) max_h_err = e;
    }
    BOOST_TEST(max_h_err < 0.06f);
}

namespace {

// Fully-wired QCfCCell over caller-owned buffers, built from float weights.
// Shared by the clamp-saturation and multi-step tests.
struct QCfCHarness
{
    static const std::size_t I = kCfcInputs, H = kCfcHidden, BB = kCfcBackbone;
    float w_bin[BB * I], w_bh[BB * H];
    float w_ff1[H * BB], w_ff2[H * BB], w_ta[H * BB], w_tb[H * BB];
    float b_bb[BB], b_ff1[H], b_ff2[H], b_a[H], b_b[H];
    int8_t w_bin_q[BB * I], w_bh_q[BB * H];
    int8_t w_ff1_q[H * BB], w_ff2_q[H * BB], w_ta_q[H * BB], w_tb_q[H * BB];
    int32_t b_bb_q[BB], b_ff1_q[H], b_ff2_q[H], b_time_q[H];
    int8_t sigmoid_lut[256], tanh_lut[256];
    float x_scale, h_scale;
    tinymind::QCfCCell<int8_t, int8_t, int32_t, int8_t, kCfcInputs, kCfcHidden, kCfcBackbone> cell;

    void build(double ts)
    {
        fillCfcWeights(w_bin, w_bh, w_ff1, w_ff2, w_ta, w_tb, b_bb, b_ff1, b_ff2, b_a, b_b);
        x_scale = 1.0f / 127.0f; h_scale = 1.0f / 127.0f;
        const float lut_scale = 8.0f / 127.0f;

        QCfCScales sc;
        sc.input_scale = x_scale; sc.hidden_scale = h_scale; sc.lut_input_scale = lut_scale;
        sc.w_backbone_input_scale  = absmax(w_bin, BB * I) / 127.0f;
        sc.w_backbone_hidden_scale = absmax(w_bh,  BB * H) / 127.0f;
        sc.w_ff1_scale = absmax(w_ff1, H * BB) / 127.0f;
        sc.w_ff2_scale = absmax(w_ff2, H * BB) / 127.0f;
        sc.w_time_a_scale = absmax(w_ta, H * BB) / 127.0f;
        sc.w_time_b_scale = absmax(w_tb, H * BB) / 127.0f;
        sc.ts = ts;

        QCfCParams pp; buildQCfCParams(sc, pp);
        quantizeQCfCBias(b_bb,  BB, lut_scale, b_bb_q);
        quantizeQCfCBias(b_ff1, H,  lut_scale, b_ff1_q);
        quantizeQCfCBias(b_ff2, H,  lut_scale, b_ff2_q);
        quantizeQCfCTimeBias(b_a, b_b, H, sc.ts, lut_scale, b_time_q);
        quantizeSymmetricWeights(w_bin, w_bin_q, BB * I, sc.w_backbone_input_scale);
        quantizeSymmetricWeights(w_bh,  w_bh_q,  BB * H, sc.w_backbone_hidden_scale);
        quantizeSymmetricWeights(w_ff1, w_ff1_q, H * BB, sc.w_ff1_scale);
        quantizeSymmetricWeights(w_ff2, w_ff2_q, H * BB, sc.w_ff2_scale);
        quantizeSymmetricWeights(w_ta,  w_ta_q,  H * BB, sc.w_time_a_scale);
        quantizeSymmetricWeights(w_tb,  w_tb_q,  H * BB, sc.w_time_b_scale);
        buildQSigmoidLUT(lut_scale, 0, 1.0f / 256.0f, -128, sigmoid_lut);
        buildQTanhLUT   (lut_scale, 0, 1.0f / 128.0f,    0, tanh_lut);

        cell.w_backbone_input = w_bin_q; cell.w_backbone_hidden = w_bh_q; cell.b_backbone = b_bb_q;
        cell.w_ff1 = w_ff1_q; cell.w_ff2 = w_ff2_q; cell.w_time_a = w_ta_q; cell.w_time_b = w_tb_q;
        cell.b_ff1 = b_ff1_q; cell.b_ff2 = b_ff2_q; cell.b_time = b_time_q;
        cell.input_zero_point = 0; cell.hidden_zero_point = 0;
        cell.backbone_input_multiplier = pp.backbone_input_multiplier;
        cell.backbone_input_shift      = pp.backbone_input_shift;
        cell.backbone_hidden_multiplier = pp.backbone_hidden_multiplier;
        cell.backbone_hidden_shift      = pp.backbone_hidden_shift;
        cell.ff1_multiplier = pp.ff1_multiplier; cell.ff1_shift = pp.ff1_shift;
        cell.ff2_multiplier = pp.ff2_multiplier; cell.ff2_shift = pp.ff2_shift;
        cell.time_a_multiplier = pp.time_a_multiplier; cell.time_a_shift = pp.time_a_shift;
        cell.time_b_multiplier = pp.time_b_multiplier; cell.time_b_shift = pp.time_b_shift;
        cell.sigmoid_lut = sigmoid_lut; cell.tanh_lut = tanh_lut;
        cell.one_minus_t_times_ff1_multiplier = pp.one_minus_t_times_ff1_multiplier;
        cell.one_minus_t_times_ff1_shift      = pp.one_minus_t_times_ff1_shift;
        cell.t_times_ff2_multiplier = pp.t_times_ff2_multiplier;
        cell.t_times_ff2_shift      = pp.t_times_ff2_shift;
        cell.output_qmin = -127; cell.output_qmax = 127;
    }
};

} // namespace

// Output clamp: a narrow [qmin, qmax] window saturates the interpolation
// result -- exercises the QCfCCell h_new clamp branches.
BOOST_AUTO_TEST_CASE(qcfc_output_clamp_saturates)
{
    QCfCHarness hz; hz.build(1.5);

    float x_ref[QCfCHarness::I] = { 0.9f, -0.9f };
    int8_t x_q[QCfCHarness::I];
    quantizeBuffer<int8_t>(x_ref, x_q, QCfCHarness::I, hz.x_scale, 0, -128, 127);

    // Unclamped reference.
    hz.cell.output_qmin = -127; hz.cell.output_qmax = 127;
    int8_t wide[QCfCHarness::H] = {0, 0, 0};
    hz.cell.forward(x_q, wide);

    int wmin = wide[0], wmax = wide[0];
    for (std::size_t i = 1; i < QCfCHarness::H; ++i)
    { if (wide[i] < wmin) wmin = wide[i]; if (wide[i] > wmax) wmax = wide[i]; }
    BOOST_TEST(wmax > wmin);   // spread required so a narrow window actually clamps

    const int8_t lo = static_cast<int8_t>(wmin + 1);
    const int8_t hi = static_cast<int8_t>(wmax - 1);
    hz.cell.output_qmin = lo; hz.cell.output_qmax = hi;
    int8_t narrow[QCfCHarness::H] = {0, 0, 0};
    hz.cell.forward(x_q, narrow);

    int changed = 0;
    for (std::size_t i = 0; i < QCfCHarness::H; ++i)
    {
        int expect = wide[i];
        if (expect < lo) expect = lo;
        if (expect > hi) expect = hi;
        BOOST_TEST(static_cast<int>(narrow[i]) == expect);
        if (narrow[i] != wide[i]) ++changed;
    }
    BOOST_TEST(changed >= 1);   // the clamp actually fired
}

// Multi-step stability: 64 recurrent steps stay int8-bounded and track a float
// CfC reference (drift bound), the QCfC analogue of the QLSTM long-sequence test.
BOOST_AUTO_TEST_CASE(qcfc_multistep_tracks_float_reference)
{
    QCfCHarness hz; hz.build(1.5);

    float x_ref[QCfCHarness::I] = { 0.5f, -0.3f };
    int8_t x_q[QCfCHarness::I];
    quantizeBuffer<int8_t>(x_ref, x_q, QCfCHarness::I, hz.x_scale, 0, -128, 127);

    float  h_f[QCfCHarness::H] = {0, 0, 0};
    int8_t h_q[QCfCHarness::H] = {0, 0, 0};
    float max_err = 0.0f;
    for (int t = 0; t < 64; ++t)
    {
        floatCfcReference(x_ref, h_f, 1.5f, hz.w_bin, hz.w_bh, hz.w_ff1, hz.w_ff2,
                          hz.w_ta, hz.w_tb, hz.b_bb, hz.b_ff1, hz.b_ff2, hz.b_a, hz.b_b);
        hz.cell.forward(x_q, h_q);
        for (std::size_t i = 0; i < QCfCHarness::H; ++i)
        {
            BOOST_TEST(h_q[i] >= -127);     // bounded, no int8 overflow
            BOOST_TEST(h_q[i] <= 127);
            const float hb = dequantize<int8_t>(h_q[i], hz.h_scale, 0);
            max_err = std::fmax(max_err, std::fabs(hb - h_f[i]));
        }
    }
    BOOST_TEST(max_err < 0.1f);    // int8 recurrent drift over 64 steps
}

// QAvgPool/QGlobalAvgPool round helper: the divide-by-zero guard returns 0
// (degenerate empty window) rather than dividing.
BOOST_AUTO_TEST_CASE(qpool_rounded_divide_zero_denominator)
{
    BOOST_TEST(tinymind::detail::roundedDivide(7, 0) == 0);
    BOOST_TEST(tinymind::detail::roundedDivide(-7, 0) == 0);
    BOOST_TEST(tinymind::detail::roundedDivide(5, 2) == 3);    // half away from zero
    BOOST_TEST(tinymind::detail::roundedDivide(-5, 2) == -3);
}

// QLayerNorm1D variance saturation: with wide (int32) input storage the per-row
// sum of squared deviations / N can exceed INT32_MAX, exercising the int64->
// int32 clamp on var_q (unreachable with the int8 deployable config).
BOOST_AUTO_TEST_CASE(qlayernorm_variance_saturates_int32)
{
    typedef tinymind::QLayerNorm1D<int32_t, int8_t, 1, 4> LN;
    int16_t gamma[4] = { 16384, 16384, 16384, 16384 };   // Q1.14 == 1.0
    int32_t beta[4]  = { 0, 0, 0, 0 };

    LN ln;
    ln.gamma = gamma; ln.beta = beta;
    ln.epsilon_q = 1;
    ln.output_multiplier = static_cast<int32_t>(1) << 30;
    ln.output_shift = 0;
    ln.output_zero_point = 0; ln.qmin = -128; ln.qmax = 127;

    // Mean 0, huge deviations: ssum/N = 1e12 > INT32_MAX -> var_q clamps.
    int32_t in[4]  = { 1000000, -1000000, 1000000, -1000000 };
    int8_t  out[4] = { 0, 0, 0, 0 };
    ln.forward(in, out);
    for (int i = 0; i < 4; ++i)
    {
        BOOST_TEST(out[i] >= -128);
        BOOST_TEST(out[i] <= 127);
    }
}

// ============================================================================
// Phase 13 -- QFFT1D, QAttention1D (linear), QAttentionSoftmax1D, QMHA.
// ============================================================================

namespace {

// Naive float DFT (no FFT) used as the bit-exact ground truth for the
// quantized FFT magnitude-spectrum parity test. O(N^2) but N is small here.
template<std::size_t N>
void naiveFloatDFT(const float* in_real, const float* in_imag,
                   float* out_real, float* out_imag)
{
    const double two_pi = 6.283185307179586476925286766559;
    for (std::size_t k = 0; k < N; ++k)
    {
        double sum_r = 0.0;
        double sum_i = 0.0;
        for (std::size_t n = 0; n < N; ++n)
        {
            const double phase = -two_pi * static_cast<double>(k) *
                                  static_cast<double>(n) /
                                  static_cast<double>(N);
            const double c = std::cos(phase);
            const double s = std::sin(phase);
            sum_r += static_cast<double>(in_real[n]) * c -
                     static_cast<double>(in_imag[n]) * s;
            sum_i += static_cast<double>(in_real[n]) * s +
                     static_cast<double>(in_imag[n]) * c;
        }
        out_real[k] = static_cast<float>(sum_r);
        out_imag[k] = static_cast<float>(sum_i);
    }
}

constexpr std::size_t kQFFTLen = 16;

// Float linear-attention (ReLU kernel) reference matching QAttention1D math.
template<std::size_t S, std::size_t E, std::size_t P>
void floatLinearAttentionReference(const float* x, const float* w,
                                   const float* b, float* y)
{
    float q[S * P], k[S * P], v[S * P], kv[P * P];

    const float* w_q = w;
    const float* w_k = w + E * P;
    const float* w_v = w + 2 * E * P;
    const float* b_q = b;
    const float* b_k = b + P;
    const float* b_v = b + 2 * P;

    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            float aq = b_q[p];
            float ak = b_k[p];
            float av = b_v[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                aq += w_q[e * P + p] * x[t * E + e];
                ak += w_k[e * P + p] * x[t * E + e];
                av += w_v[e * P + p] * x[t * E + e];
            }
            q[t * P + p] = (aq < 0.0f) ? 0.0f : aq;
            k[t * P + p] = (ak < 0.0f) ? 0.0f : ak;
            v[t * P + p] = av;
        }
    }
    for (std::size_t i = 0; i < P; ++i)
    {
        for (std::size_t j = 0; j < P; ++j)
        {
            float a = 0.0f;
            for (std::size_t t = 0; t < S; ++t)
            {
                a += k[t * P + i] * v[t * P + j];
            }
            kv[i * P + j] = a;
        }
    }
    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t j = 0; j < P; ++j)
        {
            float a = 0.0f;
            for (std::size_t i = 0; i < P; ++i)
            {
                a += q[t * P + i] * kv[i * P + j];
            }
            y[t * P + j] = a;
        }
    }
}

// Float softmax-attention reference (with 1/sqrt(P) scaling). Matches the
// QAttentionSoftmax1D math exactly.
template<std::size_t S, std::size_t E, std::size_t P>
void floatSoftmaxAttentionReference(const float* x, const float* w,
                                    const float* b, float* y)
{
    float q[S * P], k[S * P], v[S * P], scores[S * S], attn[S * S];

    const float* w_q = w;
    const float* w_k = w + E * P;
    const float* w_v = w + 2 * E * P;
    const float* b_q = b;
    const float* b_k = b + P;
    const float* b_v = b + 2 * P;

    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            float aq = b_q[p];
            float ak = b_k[p];
            float av = b_v[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                aq += w_q[e * P + p] * x[t * E + e];
                ak += w_k[e * P + p] * x[t * E + e];
                av += w_v[e * P + p] * x[t * E + e];
            }
            q[t * P + p] = aq;
            k[t * P + p] = ak;
            v[t * P + p] = av;
        }
    }
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(P));
    for (std::size_t i = 0; i < S; ++i)
    {
        for (std::size_t j = 0; j < S; ++j)
        {
            float a = 0.0f;
            for (std::size_t p = 0; p < P; ++p)
            {
                a += q[i * P + p] * k[j * P + p];
            }
            scores[i * S + j] = a * inv_sqrt;
        }
    }
    for (std::size_t i = 0; i < S; ++i)
    {
        float m = scores[i * S + 0];
        for (std::size_t j = 1; j < S; ++j)
        {
            if (scores[i * S + j] > m) m = scores[i * S + j];
        }
        float sum = 0.0f;
        for (std::size_t j = 0; j < S; ++j)
        {
            attn[i * S + j] = std::exp(scores[i * S + j] - m);
            sum += attn[i * S + j];
        }
        for (std::size_t j = 0; j < S; ++j)
        {
            attn[i * S + j] /= sum;
        }
    }
    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            float a = 0.0f;
            for (std::size_t s = 0; s < S; ++s)
            {
                a += attn[t * S + s] * v[s * P + p];
            }
            y[t * P + p] = a;
        }
    }
}

// Per-tensor symmetric quantization of a float buffer to int8.
void quantizeSymToI8(const float* src, std::size_t n,
                     float scale, int8_t* dst)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        long q = std::lround(static_cast<double>(src[i]) /
                             static_cast<double>(scale));
        if (q < -127) q = -127;
        if (q >  127) q =  127;
        dst[i] = static_cast<int8_t>(q);
    }
}

float absmaxBuf(const float* buf, std::size_t n)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        const float a = (buf[i] < 0.0f) ? -buf[i] : buf[i];
        if (a > m) m = a;
    }
    return m;
}

} // namespace

BOOST_AUTO_TEST_CASE(qfft_twiddle_table_unit_circle)
{
    // The Q1.15 twiddle factors should satisfy cos^2 + sin^2 ~= 1 at every
    // index. Tolerance accounts for the 1 LSB rounding at int16.
    int16_t cos_t[kQFFTLen / 2];
    int16_t sin_t[kQFFTLen / 2];
    buildQFFTTwiddles(kQFFTLen, cos_t, sin_t);

    BOOST_TEST(cos_t[0] == 32767);  // cos(0)
    BOOST_TEST(sin_t[0] == 0);      // sin(0)

    for (std::size_t k = 0; k < kQFFTLen / 2; ++k)
    {
        const double c = static_cast<double>(cos_t[k]) / 32768.0;
        const double s = static_cast<double>(sin_t[k]) / 32768.0;
        const double mag = c * c + s * s;
        BOOST_TEST(std::fabs(mag - 1.0) < 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(qfft_magnitude_spectrum_matches_float_reference)
{
    constexpr std::size_t N = kQFFTLen;

    // Sinusoidal input at bin 3. Real-valued (imag = 0). Amplitude 0.7 so
    // it never saturates the working int16 grid after the input scale.
    float in_real[N];
    float in_imag[N];
    const double two_pi = 6.283185307179586476925286766559;
    for (std::size_t n = 0; n < N; ++n)
    {
        in_real[n] = 0.7f * static_cast<float>(
            std::cos(two_pi * 3.0 * static_cast<double>(n) /
                     static_cast<double>(N)));
        in_imag[n] = 0.0f;
    }

    // Float DFT reference. The QFFT path scales by 1/N internally so the
    // reference uses the same convention before comparing magnitudes.
    float ref_real[N], ref_imag[N];
    naiveFloatDFT<N>(in_real, in_imag, ref_real, ref_imag);
    float ref_mag[N];
    for (std::size_t k = 0; k < N; ++k)
    {
        const float r = ref_real[k] / static_cast<float>(N);
        const float i = ref_imag[k] / static_cast<float>(N);
        ref_mag[k] = std::sqrt(r * r + i * i);
    }

    // Quantize the input to the working int16 grid. Pick the input scale so
    // the largest sample lands near +/- 30000 (well inside int16 range).
    const float work_scale = 1.0f / 30000.0f;
    int16_t work_real[N];
    int16_t work_imag[N];
    for (std::size_t n = 0; n < N; ++n)
    {
        long q = std::lround(static_cast<double>(in_real[n]) /
                             static_cast<double>(work_scale));
        if (q >  32767) q =  32767;
        if (q < -32768) q = -32768;
        work_real[n] = static_cast<int16_t>(q);
        work_imag[n] = 0;
    }

    int16_t cos_t[N / 2];
    int16_t sin_t[N / 2];
    buildQFFTTwiddles(N, cos_t, sin_t);

    tinymind::QFFT1D<N> qfft;
    qfft.twiddle_cos = cos_t;
    qfft.twiddle_sin = sin_t;
    qfft.forward(work_real, work_imag);

    // QFFT output is in (work_scale)^2 after squaring; magnitude scale is
    // work_scale. The Float DFT magnitudes were already divided by N to
    // match the scaled-butterfly convention; QFFT output is already 1/N
    // scaled, so we compare the dequantized magnitude directly.
    float qfft_mag[N];
    for (std::size_t k = 0; k < N; ++k)
    {
        const float r = static_cast<float>(work_real[k]) * work_scale;
        const float i = static_cast<float>(work_imag[k]) * work_scale;
        qfft_mag[k] = std::sqrt(r * r + i * i);
    }

    // Peak should be at bin 3 (and its conjugate bin N - 3 = 13). All other
    // bins should be small. Tolerance reflects compounded butterfly /
    // saturation noise across 4 stages at int16.
    BOOST_TEST(qfft_mag[3]  > 0.25f);
    BOOST_TEST(qfft_mag[13] > 0.25f);
    for (std::size_t k = 0; k < N; ++k)
    {
        if (k == 3 || k == 13) continue;
        BOOST_TEST(qfft_mag[k] < 0.05f);
    }

    // Bin-by-bin magnitude agreement with the float reference within a
    // generous tolerance (the scaled int16 FFT loses precision relative to
    // the float DFT especially in the noise floor).
    float max_err = 0.0f;
    for (std::size_t k = 0; k < N; ++k)
    {
        const float e = std::fabs(qfft_mag[k] - ref_mag[k]);
        if (e > max_err) max_err = e;
    }
    BOOST_TEST(max_err < 0.05f);
}

BOOST_AUTO_TEST_CASE(qfft_forward_inverse_round_trip_is_close_to_identity)
{
    constexpr std::size_t N = kQFFTLen;
    int16_t real[N];
    int16_t imag[N];
    for (std::size_t n = 0; n < N; ++n)
    {
        // Mid-amplitude triangular pattern.
        real[n] = static_cast<int16_t>(1000 * static_cast<int32_t>(
                    static_cast<int32_t>(n) - static_cast<int32_t>(N / 2)));
        imag[n] = 0;
    }
    int16_t saved_real[N];
    for (std::size_t n = 0; n < N; ++n) saved_real[n] = real[n];

    int16_t cos_t[N / 2];
    int16_t sin_t[N / 2];
    buildQFFTTwiddles(N, cos_t, sin_t);

    tinymind::QFFT1D<N> qfft;
    qfft.twiddle_cos = cos_t;
    qfft.twiddle_sin = sin_t;
    qfft.forward(real, imag);
    qfft.inverse(real, imag);

    // Each butterfly stage rounds, so a tight identity is not achievable in
    // int16 -- check that the inverse recovers the input shape up to a few
    // hundred LSBs of noise.
    int32_t max_err = 0;
    for (std::size_t n = 0; n < N; ++n)
    {
        const int32_t e = static_cast<int32_t>(real[n]) -
                          static_cast<int32_t>(saved_real[n]);
        const int32_t a = (e < 0) ? -e : e;
        if (a > max_err) max_err = a;
    }
    BOOST_TEST(max_err < 500);
}

BOOST_AUTO_TEST_CASE(qattention_linear_parity_with_float_reference)
{
    constexpr std::size_t S = 4;
    constexpr std::size_t E = 6;
    constexpr std::size_t P = 4;

    // Reproducible non-trivial inputs and weights.
    float x[S * E];
    for (std::size_t i = 0; i < S * E; ++i)
    {
        x[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 7) - 3);
    }
    float w[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
    {
        w[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 11) - 5);
    }
    float b[3 * P];
    for (std::size_t i = 0; i < 3 * P; ++i)
    {
        b[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 3) - 1);
    }

    float y_ref[S * P];
    floatLinearAttentionReference<S, E, P>(x, w, b, y_ref);

    // Calibrate. Use symmetric weights, asymmetric activations (zero_point = 0
    // is the symmetric special case used throughout the Q* tests).
    const float x_scale = absmaxBuf(x, S * E) / 127.0f;
    const float wq_scale = absmaxBuf(w,             E * P) / 127.0f;
    const float wk_scale = absmaxBuf(w + E * P,     E * P) / 127.0f;
    const float wv_scale = absmaxBuf(w + 2 * E * P, E * P) / 127.0f;

    // Run the float reference once more, capturing per-tensor ranges of Q,
    // K, V, KV so we can calibrate output scales for each.
    float q_buf[S * P], k_buf[S * P], v_buf[S * P], kv_buf[P * P];
    {
        const float* wq = w;
        const float* wk = w + E * P;
        const float* wv = w + 2 * E * P;
        const float* bq = b;
        const float* bk = b + P;
        const float* bv = b + 2 * P;
        for (std::size_t t = 0; t < S; ++t)
        {
            for (std::size_t p = 0; p < P; ++p)
            {
                float aq = bq[p], ak = bk[p], av = bv[p];
                for (std::size_t e = 0; e < E; ++e)
                {
                    aq += wq[e * P + p] * x[t * E + e];
                    ak += wk[e * P + p] * x[t * E + e];
                    av += wv[e * P + p] * x[t * E + e];
                }
                q_buf[t * P + p] = (aq < 0.0f) ? 0.0f : aq;
                k_buf[t * P + p] = (ak < 0.0f) ? 0.0f : ak;
                v_buf[t * P + p] = av;
            }
        }
        for (std::size_t i = 0; i < P; ++i)
        {
            for (std::size_t j = 0; j < P; ++j)
            {
                float a = 0.0f;
                for (std::size_t t = 0; t < S; ++t)
                {
                    a += k_buf[t * P + i] * v_buf[t * P + j];
                }
                kv_buf[i * P + j] = a;
            }
        }
    }

    const float q_scale  = absmaxBuf(q_buf,  S * P) / 127.0f;
    const float k_scale  = absmaxBuf(k_buf,  S * P) / 127.0f;
    const float v_scale  = absmaxBuf(v_buf,  S * P) / 127.0f;
    const float kv_scale = absmaxBuf(kv_buf, P * P) / 127.0f;
    const float y_scale  = absmaxBuf(y_ref,  S * P) / 127.0f;

    // Quantize tensors.
    int8_t x_q[S * E];
    int8_t w_q[3 * E * P];
    int32_t b_q[3 * P];
    quantizeBuffer<int8_t>(x, x_q, S * E, x_scale, 0, -128, 127);
    quantizeSymToI8(w,             E * P, wq_scale, w_q);
    quantizeSymToI8(w + E * P,     E * P, wk_scale, w_q + E * P);
    quantizeSymToI8(w + 2 * E * P, E * P, wv_scale, w_q + 2 * E * P);

    // Per-projection bias is int32 in (input_scale * weight_scale) units.
    for (std::size_t p = 0; p < P; ++p)
    {
        b_q[p] = static_cast<int32_t>(std::lround(
            static_cast<double>(b[p]) /
            (static_cast<double>(x_scale) * static_cast<double>(wq_scale))));
        b_q[P + p] = static_cast<int32_t>(std::lround(
            static_cast<double>(b[P + p]) /
            (static_cast<double>(x_scale) * static_cast<double>(wk_scale))));
        b_q[2 * P + p] = static_cast<int32_t>(std::lround(
            static_cast<double>(b[2 * P + p]) /
            (static_cast<double>(x_scale) * static_cast<double>(wv_scale))));
    }

    tinymind::QAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
                           S, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.input_zero_point = 0;
    attn.q_zero_point     = 0;
    attn.k_zero_point     = 0;
    attn.v_zero_point     = 0;
    attn.kv_zero_point    = 0;

    attn.q_requantizer = buildRequantizer<int8_t>(x_scale, wq_scale, q_scale,
                                                  0, 0, 127);    // ReLU folded: qmin = zp
    attn.k_requantizer = buildRequantizer<int8_t>(x_scale, wk_scale, k_scale,
                                                  0, 0, 127);
    attn.v_requantizer = buildRequantizer<int8_t>(x_scale, wv_scale, v_scale,
                                                  0, -128, 127);
    attn.kv_requantizer = buildRequantizer<int8_t>(k_scale, v_scale, kv_scale,
                                                   0, -128, 127);
    attn.output_requantizer = buildRequantizer<int8_t>(q_scale, kv_scale,
                                                       y_scale, 0, -128, 127);

    int8_t q_sc[S * P], k_sc[S * P], v_sc[S * P], kv_sc[P * P];
    int8_t y_q[S * P];
    attn.forward(x_q, q_sc, k_sc, v_sc, kv_sc, y_q);

    const float y_ref_max = absmaxBuf(y_ref, S * P);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < S * P; ++i)
    {
        const float y_back = dequantize<int8_t>(y_q[i], y_scale, 0);
        const float e = std::fabs(y_back - y_ref[i]);
        if (e > max_err) max_err = e;
    }

    // The end-to-end MAC stack (three projection requantizers + KV
    // requantizer + output requantizer) compounds at int8. A few percent of
    // the output dynamic range is the empirical noise floor.
    BOOST_TEST(max_err < 0.08f * y_ref_max);
}

BOOST_AUTO_TEST_CASE(qattention_softmax_parity_with_float_reference)
{
    constexpr std::size_t S = 4;
    constexpr std::size_t E = 6;
    constexpr std::size_t P = 4;

    float x[S * E];
    for (std::size_t i = 0; i < S * E; ++i)
    {
        x[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 5) - 2);
    }
    float w[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
    {
        w[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 9) - 4);
    }
    float b[3 * P];
    for (std::size_t i = 0; i < 3 * P; ++i)
    {
        b[i] = 0.0f;
    }

    float y_ref[S * P];
    floatSoftmaxAttentionReference<S, E, P>(x, w, b, y_ref);

    // Run the projections once more for output-tensor range observation.
    float q_buf[S * P], k_buf[S * P], v_buf[S * P], scores_buf[S * S];
    {
        const float* wq = w;
        const float* wk = w + E * P;
        const float* wv = w + 2 * E * P;
        for (std::size_t t = 0; t < S; ++t)
        {
            for (std::size_t p = 0; p < P; ++p)
            {
                float aq = 0.0f, ak = 0.0f, av = 0.0f;
                for (std::size_t e = 0; e < E; ++e)
                {
                    aq += wq[e * P + p] * x[t * E + e];
                    ak += wk[e * P + p] * x[t * E + e];
                    av += wv[e * P + p] * x[t * E + e];
                }
                q_buf[t * P + p] = aq;
                k_buf[t * P + p] = ak;
                v_buf[t * P + p] = av;
            }
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(P));
        for (std::size_t i = 0; i < S; ++i)
        {
            for (std::size_t j = 0; j < S; ++j)
            {
                float a = 0.0f;
                for (std::size_t p = 0; p < P; ++p)
                {
                    a += q_buf[i * P + p] * k_buf[j * P + p];
                }
                scores_buf[i * S + j] = a * inv;
            }
        }
    }

    const float x_scale  = absmaxBuf(x, S * E) / 127.0f;
    const float wq_scale = absmaxBuf(w,             E * P) / 127.0f;
    const float wk_scale = absmaxBuf(w + E * P,     E * P) / 127.0f;
    const float wv_scale = absmaxBuf(w + 2 * E * P, E * P) / 127.0f;
    const float q_scale  = absmaxBuf(q_buf, S * P) / 127.0f;
    const float k_scale  = absmaxBuf(k_buf, S * P) / 127.0f;
    const float v_scale  = absmaxBuf(v_buf, S * P) / 127.0f;
    const float score_scale = absmaxBuf(scores_buf, S * S) / 127.0f;
    const float attn_scale  = 1.0f / 256.0f;    // TFLite softmax convention
    const float y_scale  = absmaxBuf(y_ref, S * P) / 127.0f;

    int8_t x_q[S * E];
    int8_t w_q[3 * E * P];
    int32_t b_q[3 * P] = {0};
    quantizeBuffer<int8_t>(x, x_q, S * E, x_scale, 0, -128, 127);
    quantizeSymToI8(w,             E * P, wq_scale, w_q);
    quantizeSymToI8(w + E * P,     E * P, wk_scale, w_q + E * P);
    quantizeSymToI8(w + 2 * E * P, E * P, wv_scale, w_q + 2 * E * P);

    int32_t exp_lut[256];
    buildQSoftmaxExpLUT(score_scale, exp_lut);

    tinymind::QAttentionSoftmax1D<int8_t, int8_t, int32_t,
                                  int8_t, int8_t, int8_t, int8_t,
                                  S, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.input_zero_point = 0;
    attn.q_zero_point     = 0;
    attn.k_zero_point     = 0;
    attn.v_zero_point     = 0;
    attn.attn_zero_point  = -128;

    attn.q_requantizer = buildRequantizer<int8_t>(x_scale, wq_scale, q_scale,
                                                  0, -128, 127);
    attn.k_requantizer = buildRequantizer<int8_t>(x_scale, wk_scale, k_scale,
                                                  0, -128, 127);
    attn.v_requantizer = buildRequantizer<int8_t>(x_scale, wv_scale, v_scale,
                                                  0, -128, 127);
    // Score requantizer folds the 1 / sqrt(P) attention factor.
    {
        Requantizer<int32_t, int8_t> r;
        const double ratio =
            (static_cast<double>(q_scale) * static_cast<double>(k_scale)) /
            static_cast<double>(score_scale) *
            qAttentionInvSqrt(P);
        int32_t mult = 0, shft = 0;
        quantizeMultiplier(ratio, mult, shft);
        r.multiplier = mult;
        r.shift = shft;
        r.zero_point = 0;
        r.qmin = -128;
        r.qmax = 127;
        attn.score_requantizer = r;
    }
    attn.softmax_exp_lut = exp_lut;
    attn.attn_qmin = -128;
    attn.attn_qmax = 127;

    attn.output_requantizer = buildRequantizer<int8_t>(attn_scale, v_scale,
                                                       y_scale, 0, -128, 127);

    int8_t q_sc[S * P], k_sc[S * P], v_sc[S * P];
    int8_t score_sc[S * S], attn_sc[S * S];
    int8_t y_q[S * P];
    attn.forward(x_q, q_sc, k_sc, v_sc, score_sc, attn_sc, y_q);

    const float y_ref_max = absmaxBuf(y_ref, S * P);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < S * P; ++i)
    {
        const float y_back = dequantize<int8_t>(y_q[i], y_scale, 0);
        const float e = std::fabs(y_back - y_ref[i]);
        if (e > max_err) max_err = e;
    }
    // Softmax attention compounds more stages than linear attention; the
    // empirical noise floor is ~15% of dynamic range for these tiny shapes.
    BOOST_TEST(max_err < 0.15f * y_ref_max);
}

BOOST_AUTO_TEST_CASE(qattention_softmax_zero_lut_clamps_attention_row)
{
    // Zero weights -> zero scores -> an all-zero softmax exp LUT makes every
    // score-row sum 0. The forward must take the divide-by-zero guard and
    // fill each attention row with the clamped attn_zero_point.
    constexpr std::size_t S = 2, E = 2, P = 2;

    int8_t w_q[3 * E * P] = {0};
    int32_t b_q[3 * P]    = {0};
    int32_t exp_lut[256]  = {0};

    tinymind::QAttentionSoftmax1D<int8_t, int8_t, int32_t,
                                  int8_t, int8_t, int8_t, int8_t,
                                  S, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.input_zero_point = 0;
    attn.q_zero_point = 0;
    attn.k_zero_point = 0;
    attn.v_zero_point = 0;
    attn.attn_zero_point = -128;
    attn.q_requantizer = makeI8Requant(kQ31One, 0, 0);
    attn.k_requantizer = makeI8Requant(kQ31One, 0, 0);
    attn.v_requantizer = makeI8Requant(kQ31One, 0, 0);
    attn.score_requantizer = makeI8Requant(kQ31One, 0, 0);
    attn.softmax_exp_lut = exp_lut;
    attn.attn_qmin = -128;
    attn.attn_qmax = 127;
    attn.output_requantizer = makeI8Requant(kQ31One, 0, 0);

    int8_t x_q[S * E] = {5, -5, 10, -10};
    int8_t q_sc[S * P], k_sc[S * P], v_sc[S * P];
    int8_t score_sc[S * S], attn_sc[S * S] = {1, 1, 1, 1};
    int8_t y_q[S * P];
    attn.forward(x_q, q_sc, k_sc, v_sc, score_sc, attn_sc, y_q);

    for (std::size_t i = 0; i < S * S; ++i)
    {
        BOOST_TEST(static_cast<int>(attn_sc[i]) == -128);
    }
}

BOOST_AUTO_TEST_CASE(qmha_stacks_two_identical_heads)
{
    // Two identical heads should each emit the same per-head output as
    // their underlying single-head layer; the wrapper stacks them along
    // the projection axis. Verifies the stack layout, not the math.
    constexpr std::size_t S = 3;
    constexpr std::size_t E = 4;
    constexpr std::size_t P = 2;
    constexpr std::size_t H = 2;

    typedef tinymind::QAttention1D<int8_t, int8_t, int32_t, int8_t,
                                   int8_t, int8_t, S, E, P> HeadT;

    int8_t w_q[3 * E * P] = {0};
    int32_t b_q[3 * P]    = {0};
    // Identity-ish weight pattern; not testing math, only the stack.
    for (std::size_t e = 0; e < E; ++e)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            w_q[e * P + p]             = static_cast<int8_t>((e + p) & 0x7F);
            w_q[E * P + e * P + p]     = static_cast<int8_t>((e * p) & 0x7F);
            w_q[2 * E * P + e * P + p] = static_cast<int8_t>((e + 1) & 0x7F);
        }
    }

    auto r = makeI8Requant(kQ31One, 16, 0);

    HeadT head;
    head.weights = w_q;
    head.biases  = b_q;
    head.input_zero_point = 0;
    head.q_zero_point     = 0;
    head.k_zero_point     = 0;
    head.v_zero_point     = 0;
    head.kv_zero_point    = 0;
    head.q_requantizer = r;  head.q_requantizer.qmin = 0;
    head.k_requantizer = r;  head.k_requantizer.qmin = 0;
    head.v_requantizer = r;
    head.kv_requantizer = r;
    head.output_requantizer = r;

    int8_t x_q[S * E];
    for (std::size_t i = 0; i < S * E; ++i) x_q[i] = static_cast<int8_t>(i + 1);

    int8_t single_q[HeadT::QScratchSize];
    int8_t single_k[HeadT::KScratchSize];
    int8_t single_v[HeadT::VScratchSize];
    int8_t single_kv[HeadT::KVScratchSize];
    int8_t single_out[HeadT::OutputSize];
    head.forward(x_q, single_q, single_k, single_v, single_kv, single_out);

    tinymind::QMultiHeadLinearAttention1D<
        int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
        S, E, P, H> mha;
    mha.heads[0] = head;
    mha.heads[1] = head;

    int8_t mha_q[HeadT::QScratchSize];
    int8_t mha_k[HeadT::KScratchSize];
    int8_t mha_v[HeadT::VScratchSize];
    int8_t mha_kv[HeadT::KVScratchSize];
    int8_t mha_head_out[S * P];
    int8_t mha_out[S * P * H];
    mha.forward(x_q, mha_q, mha_k, mha_v, mha_kv, mha_head_out, mha_out);

    // Both head slices in the stacked output should match the single-head
    // output byte-for-byte.
    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            for (std::size_t h = 0; h < H; ++h)
            {
                const int8_t got = mha_out[t * (P * H) + h * P + p];
                const int8_t exp = single_out[t * P + p];
                BOOST_TEST(static_cast<int>(got) == static_cast<int>(exp));
            }
        }
    }
}

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

// ---------------------------------------------------------------------------
// Phase 14: SIMD dispatch bit-exactness.
//
// Whichever TINYMIND_ENABLE_SIMD_* gate is on at build time, the dispatched
// int8 dot product must match the scalar reference bit-for-bit across
// pathological inputs: aligned vector lengths, mid-vector tails, sign-mix
// of weights and inputs, edge-of-int8-range values, and zero_point != 0.
// The same invariant carries up into QDense and QConv2D since both call
// into the dispatch helper for the inner reduction.
// ---------------------------------------------------------------------------

namespace simd_test {

    inline int32_t scalarDot(const int8_t* x, const int8_t* w,
                             std::size_t n, int8_t zp)
    {
        return tinymind::simd::int8DotWithZeroPointScalar(x, w, n, zp);
    }

    inline int32_t dispatchedDot(const int8_t* x, const int8_t* w,
                                 std::size_t n, int8_t zp)
    {
        return tinymind::simd::int8DotWithZeroPoint(x, w, n, zp);
    }

    // Deterministic fill — same seed every run keeps the bit-exactness
    // check reproducible across builds.
    inline void fillSawtooth(int8_t* p, std::size_t n, int seed)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            const int v = static_cast<int>((i * 7 + seed * 13) & 0xFF) - 128;
            p[i] = static_cast<int8_t>(v);
        }
    }

}

BOOST_AUTO_TEST_CASE(simd_int8_dot_bit_exact_across_lengths)
{
    // Lengths span: short scalar-tail-only, vector-aligned, vector + 1,
    // long contiguous, and odd primes that hit every tail residue.
    const std::size_t lengths[] = { 0, 1, 7, 15, 16, 17, 31, 32, 33,
                                    63, 64, 65, 127, 128, 129, 257, 1024 };
    int8_t x[1024];
    int8_t w[1024];

    for (std::size_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); ++li)
    {
        const std::size_t n = lengths[li];
        simd_test::fillSawtooth(x, n, 11);
        simd_test::fillSawtooth(w, n, 47);

        // Sweep zero_point over the full asymmetric int8 range so the
        // bias-correction term (zp * sum_w) is exercised at every bit.
        for (int zp = -128; zp <= 127; zp += 31)
        {
            const int32_t a = simd_test::scalarDot(x, w, n,
                                                   static_cast<int8_t>(zp));
            const int32_t b = simd_test::dispatchedDot(x, w, n,
                                                       static_cast<int8_t>(zp));
            BOOST_TEST(a == b);
        }
    }
}

BOOST_AUTO_TEST_CASE(simd_int8_dot_extreme_values_bit_exact)
{
    // INT8_MIN x INT8_MIN, INT8_MAX x INT8_MAX, mixed signs — saturate-
    // sensitive backends (PMADDUBSW chain on AVX2 without the cvtepi8
    // widening) would fail this; the chosen implementation does not.
    int8_t x[128];
    int8_t w[128];
    for (std::size_t i = 0; i < 128; ++i)
    {
        x[i] = (i & 1) ? static_cast<int8_t>(127)
                       : static_cast<int8_t>(-128);
        w[i] = (i & 2) ? static_cast<int8_t>(127)
                       : static_cast<int8_t>(-128);
    }
    for (int zp : { -128, -1, 0, 1, 127 })
    {
        const int32_t a = simd_test::scalarDot(x, w, 128,
                                               static_cast<int8_t>(zp));
        const int32_t b = simd_test::dispatchedDot(x, w, 128,
                                                   static_cast<int8_t>(zp));
        BOOST_TEST(a == b);
    }
}

BOOST_AUTO_TEST_CASE(simd_qdense_bit_exact_vs_scalar_reference)
{
    // QDense calls into the dispatch helper for every output row. Build a
    // reference by re-doing the math with the scalar primitive and compare
    // the int8 output element-by-element.
    constexpr std::size_t In  = 200;
    constexpr std::size_t Out = 7;
    QDense<int8_t, int8_t, int32_t, int8_t, In, Out> dense;

    int8_t w[In * Out];
    int32_t b[Out];
    int8_t x[In];
    int8_t y[Out];

    simd_test::fillSawtooth(x, In, 3);
    for (std::size_t o = 0; o < Out; ++o)
    {
        simd_test::fillSawtooth(w + o * In, In, static_cast<int>(o) * 5 + 1);
        b[o] = static_cast<int32_t>(o) * 64 - 96;
    }

    Requantizer<int32_t, int8_t> r;
    r.multiplier = static_cast<int32_t>(1) << 30;
    r.shift = 6;
    r.zero_point = -2;
    r.qmin = -128;
    r.qmax = 127;

    dense.weights = w;
    dense.biases = b;
    dense.input_zero_point = static_cast<int8_t>(11);
    dense.requantizer = r;
    dense.forward(x, y);

    for (std::size_t o = 0; o < Out; ++o)
    {
        int32_t acc = b[o] +
            simd_test::scalarDot(x, w + o * In, In,
                                 static_cast<int8_t>(11));
        const int8_t expected = r.apply(acc);
        BOOST_TEST(static_cast<int>(y[o]) == static_cast<int>(expected));
    }
}

BOOST_AUTO_TEST_CASE(simd_qconv2d_bit_exact_vs_scalar_reference)
{
    // Small but realistic conv shape: 6x6 input, 3x3 kernel, stride 1,
    // 4 in-channels, 5 filters. InChannels=4 exercises the SIMD tail
    // (most backends operate on >=8 or >=16 lanes); the kernel loop
    // amortizes many short-tail invocations.
    constexpr std::size_t H = 6, W = 6, IC = 4, KH = 3, KW = 3, F = 5;
    QConv2D<int8_t, int8_t, int32_t, int8_t, H, W, IC, KH, KW, 1, 1, F>
        conv;

    constexpr std::size_t TotalW = F * KH * KW * IC;
    constexpr std::size_t OutH = (H - KH) + 1;
    constexpr std::size_t OutW = (W - KW) + 1;

    int8_t w[TotalW];
    int32_t b[F];
    int8_t x[H * W * IC];
    int8_t y[OutH * OutW * F];

    simd_test::fillSawtooth(x, H * W * IC, 9);
    simd_test::fillSawtooth(w, TotalW, 21);
    for (std::size_t f = 0; f < F; ++f)
    {
        b[f] = static_cast<int32_t>(f) * 17 - 30;
    }

    Requantizer<int32_t, int8_t> r;
    r.multiplier = static_cast<int32_t>(1) << 29;
    r.shift = 5;
    r.zero_point = 3;
    r.qmin = -128;
    r.qmax = 127;

    conv.weights = w;
    conv.biases = b;
    conv.input_zero_point = static_cast<int8_t>(-7);
    conv.requantizer = r;
    conv.forward(x, y);

    // Scalar reference: recompute everything via simd_test::scalarDot and
    // compare element-by-element.
    for (std::size_t oh = 0; oh < OutH; ++oh)
    {
        for (std::size_t ow = 0; ow < OutW; ++ow)
        {
            for (std::size_t f = 0; f < F; ++f)
            {
                int32_t acc = b[f];
                for (std::size_t kh = 0; kh < KH; ++kh)
                {
                    for (std::size_t kw = 0; kw < KW; ++kw)
                    {
                        const std::size_t in_off =
                            ((oh + kh) * W + (ow + kw)) * IC;
                        const std::size_t w_off =
                            (f * KH * KW + kh * KW + kw) * IC;
                        acc += simd_test::scalarDot(x + in_off, w + w_off,
                                                    IC,
                                                    static_cast<int8_t>(-7));
                    }
                }
                const int8_t expected = r.apply(acc);
                const std::size_t out_idx = (oh * OutW + ow) * F + f;
                BOOST_TEST(static_cast<int>(y[out_idx]) ==
                           static_cast<int>(expected));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(simd_active_backend_name_reports_a_known_value)
{
    // The dispatch reports which backend it resolved to. The exact value
    // depends on the build's TINYMIND_ENABLE_SIMD_* flags, but it must be
    // one of the documented names.
    const char* const name = tinymind::simd::activeBackendName();
    BOOST_TEST(name != nullptr);
    static const char* const kKnown[] = {
        "scalar", "avx2", "avx_vnni", "avx512f", "avx512_vnni",
        "neon", "neon_dotprod", "sve", "helium_mve_i"
    };
    bool found = false;
    for (size_t i = 0; i < sizeof(kKnown) / sizeof(kKnown[0]); ++i)
    {
        // string compare without pulling in <cstring> stylistically; the
        // test fixture already includes <cstring> transitively via Boost.
        const char* a = name;
        const char* b = kKnown[i];
        while (*a && (*a == *b)) { ++a; ++b; }
        if (*a == *b) { found = true; break; }
    }
    BOOST_TEST(found);
}

// ---------------------------------------------------------------------------
// QEmbedding -- token gather (transformer input layer).
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(qembedding_gather_selects_correct_rows)
{
    constexpr std::size_t V = 4, E = 3, S = 3;
    const int8_t table[V * E] =
    {
         10,  11,  12,   // id 0
         -5,  -6,  -7,   // id 1
         20,  21,  22,   // id 2
          1,   2,   3,   // id 3
    };

    tinymind::QEmbedding<int8_t, int8_t, V, E> emb;
    emb.table = table;
    emb.requantizer = nullptr;

    const int32_t tokens[S] = {2, 0, 3};
    int8_t out[S * E];
    emb.forward(tokens, S, out);

    for (std::size_t t = 0; t < S; ++t)
    {
        const std::size_t id = static_cast<std::size_t>(tokens[t]);
        for (std::size_t e = 0; e < E; ++e)
        {
            BOOST_TEST(static_cast<int>(out[t * E + e]) ==
                       static_cast<int>(table[id * E + e]));
        }
    }
}

BOOST_AUTO_TEST_CASE(qembedding_requantizer_rescales_gathered_codes)
{
    constexpr std::size_t V = 2, E = 4, S = 2;
    const int8_t table[V * E] =
    {
         10, -6, 0, 20,
        -12,  5, 1, -3,
    };

    // ratio = (input_scale * weight_scale) / output_scale = (2 * 1) / 1 = 2:
    // a power-of-two multiplier doubles each code exactly (no rounding noise).
    auto r = buildRequantizer<int8_t>(2.0f, 1.0f, 1.0f, 0, -128, 127);

    tinymind::QEmbedding<int8_t, int8_t, V, E> emb;
    emb.table = table;
    emb.requantizer = &r;

    const int32_t tokens[S] = {1, 0};
    int8_t out[S * E];
    emb.forward(tokens, S, out);

    for (std::size_t t = 0; t < S; ++t)
    {
        const std::size_t id = static_cast<std::size_t>(tokens[t]);
        for (std::size_t e = 0; e < E; ++e)
        {
            BOOST_TEST(static_cast<int>(out[t * E + e]) ==
                       2 * static_cast<int>(table[id * E + e]));
        }
    }
}

// ---------------------------------------------------------------------------
// QPositionalEncoding1D -- fused positional add + sinusoidal table generator.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(qpositional_add_matches_float_reference)
{
    constexpr std::size_t S = 2, E = 3, N = S * E;

    const float in_f[N]  = {-0.4f, 0.9f, 0.1f, -1.0f, 0.5f, 0.7f};
    const float tab_f[N] = { 0.2f, -0.3f, 0.6f, 0.4f, -0.8f, 0.1f};

    const float scale_in = 0.01f;  const int32_t zp_in = -3;
    const float scale_tab = 0.008f; const int32_t zp_tab = 5;
    const float scale_out = 0.015f; const int32_t zp_out = -2;

    int8_t qin[N], qtab[N], out[N];
    quantizeBuffer<int8_t>(in_f, qin, N, scale_in, zp_in, -128, 127);
    quantizeBuffer<int8_t>(tab_f, qtab, N, scale_tab, zp_tab, -128, 127);

    tinymind::QPositionalEncoding1D<int8_t, int8_t, int8_t, S, E> pos;
    pos.table = qtab;
    const QAddParams p = buildQAddParams(scale_in, scale_tab, scale_out);
    pos.adder.input_a_zero_point = static_cast<int8_t>(zp_in);
    pos.adder.input_b_zero_point = static_cast<int8_t>(zp_tab);
    pos.adder.left_shift = p.left_shift;
    pos.adder.input_a_multiplier = p.input_a_multiplier;
    pos.adder.input_a_shift = p.input_a_shift;
    pos.adder.input_b_multiplier = p.input_b_multiplier;
    pos.adder.input_b_shift = p.input_b_shift;
    pos.adder.output_requantizer.multiplier = p.output_multiplier;
    pos.adder.output_requantizer.shift = p.output_shift;
    pos.adder.output_requantizer.zero_point = static_cast<int8_t>(zp_out);
    pos.adder.output_requantizer.qmin = -128;
    pos.adder.output_requantizer.qmax = 127;

    pos.forward(qin, out);

    for (std::size_t i = 0; i < N; ++i)
    {
        const float deq = dequantize<int8_t>(out[i], scale_out, zp_out);
        const float ref = in_f[i] + tab_f[i];
        BOOST_TEST(std::abs(deq - ref) < scale_out + 1e-3f);
    }
}

BOOST_AUTO_TEST_CASE(qpositional_sinusoidal_table_matches_formula)
{
    constexpr std::size_t S = 4, D = 4;
    float t[S * D];
    tinymind::sinusoidalPositionalTable(S, D, t);

    // pos 0: even index sin(0) = 0, odd index cos(0) = 1.
    BOOST_TEST(std::fabs(t[0] - 0.0f) < 1e-6f);
    BOOST_TEST(std::fabs(t[1] - 1.0f) < 1e-6f);
    BOOST_TEST(std::fabs(t[2] - 0.0f) < 1e-6f);
    BOOST_TEST(std::fabs(t[3] - 1.0f) < 1e-6f);

    // pos 1: pair 0 -> denom 1 -> angle 1; pair 1 -> denom 10000^0.5=100 ->
    // angle 0.01.
    BOOST_TEST(std::fabs(t[1 * D + 0] - std::sin(1.0f)) < 1e-5f);
    BOOST_TEST(std::fabs(t[1 * D + 1] - std::cos(1.0f)) < 1e-5f);
    BOOST_TEST(std::fabs(t[1 * D + 2] - std::sin(0.01f)) < 1e-5f);
    BOOST_TEST(std::fabs(t[1 * D + 3] - std::cos(0.01f)) < 1e-5f);
}

// ---------------------------------------------------------------------------
// Decoder attention float references (causal self + cross). Placed in an
// anonymous namespace so they stay file-local like the bidirectional refs.
// ---------------------------------------------------------------------------
namespace {

// Causal linear (ReLU-kernel) attention: y[t] = q'[t] . (sum_{s<=t} k'[s]^T v[s]).
template<std::size_t S, std::size_t E, std::size_t P>
void floatCausalLinearAttentionReference(const float* x, const float* w,
                                         const float* b, float* y)
{
    float q[S * P], k[S * P], v[S * P];
    const float* w_q = w;
    const float* w_k = w + E * P;
    const float* w_v = w + 2 * E * P;
    const float* b_q = b;
    const float* b_k = b + P;
    const float* b_v = b + 2 * P;

    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            float aq = b_q[p], ak = b_k[p], av = b_v[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                aq += w_q[e * P + p] * x[t * E + e];
                ak += w_k[e * P + p] * x[t * E + e];
                av += w_v[e * P + p] * x[t * E + e];
            }
            q[t * P + p] = (aq < 0.0f) ? 0.0f : aq;
            k[t * P + p] = (ak < 0.0f) ? 0.0f : ak;
            v[t * P + p] = av;
        }
    }

    float kv[P * P];
    for (std::size_t i = 0; i < P * P; ++i) kv[i] = 0.0f;
    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t i = 0; i < P; ++i)
        {
            for (std::size_t j = 0; j < P; ++j)
            {
                kv[i * P + j] += k[t * P + i] * v[t * P + j];
            }
        }
        for (std::size_t j = 0; j < P; ++j)
        {
            float a = 0.0f;
            for (std::size_t i = 0; i < P; ++i)
            {
                a += q[t * P + i] * kv[i * P + j];
            }
            y[t * P + j] = a;
        }
    }
}

// Causal softmax attention: query t attends to keys/values j <= t.
template<std::size_t S, std::size_t E, std::size_t P>
void floatCausalSoftmaxAttentionReference(const float* x, const float* w,
                                          const float* b, float* y)
{
    float q[S * P], k[S * P], v[S * P];
    const float* w_q = w;
    const float* w_k = w + E * P;
    const float* w_v = w + 2 * E * P;
    const float* b_q = b;
    const float* b_k = b + P;
    const float* b_v = b + 2 * P;

    for (std::size_t t = 0; t < S; ++t)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            float aq = b_q[p], ak = b_k[p], av = b_v[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                aq += w_q[e * P + p] * x[t * E + e];
                ak += w_k[e * P + p] * x[t * E + e];
                av += w_v[e * P + p] * x[t * E + e];
            }
            q[t * P + p] = aq;
            k[t * P + p] = ak;
            v[t * P + p] = av;
        }
    }

    const float inv = 1.0f / std::sqrt(static_cast<float>(P));
    for (std::size_t t = 0; t < S; ++t)
    {
        float sc[S];
        float m = -1.0e30f;
        for (std::size_t j = 0; j <= t; ++j)
        {
            float a = 0.0f;
            for (std::size_t p = 0; p < P; ++p)
            {
                a += q[t * P + p] * k[j * P + p];
            }
            sc[j] = a * inv;
            if (sc[j] > m) m = sc[j];
        }
        float sum = 0.0f;
        for (std::size_t j = 0; j <= t; ++j)
        {
            sc[j] = std::exp(sc[j] - m);
            sum += sc[j];
        }
        for (std::size_t p = 0; p < P; ++p)
        {
            float a = 0.0f;
            for (std::size_t j = 0; j <= t; ++j)
            {
                a += (sc[j] / sum) * v[j * P + p];
            }
            y[t * P + p] = a;
        }
    }
}

// Linear cross-attention: Q' from decoder, K'/V from encoder memory.
template<std::size_t SDec, std::size_t SEnc, std::size_t E, std::size_t P>
void floatCrossLinearReference(const float* dec, const float* mem,
                               const float* w, const float* b, float* y)
{
    const float* w_q = w;
    const float* w_k = w + E * P;
    const float* w_v = w + 2 * E * P;
    const float* b_q = b;
    const float* b_k = b + P;
    const float* b_v = b + 2 * P;

    float k[SEnc * P], v[SEnc * P];
    for (std::size_t s = 0; s < SEnc; ++s)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            float ak = b_k[p], av = b_v[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                ak += w_k[e * P + p] * mem[s * E + e];
                av += w_v[e * P + p] * mem[s * E + e];
            }
            k[s * P + p] = (ak < 0.0f) ? 0.0f : ak;
            v[s * P + p] = av;
        }
    }
    float kv[P * P];
    for (std::size_t i = 0; i < P * P; ++i) kv[i] = 0.0f;
    for (std::size_t i = 0; i < P; ++i)
    {
        for (std::size_t j = 0; j < P; ++j)
        {
            for (std::size_t s = 0; s < SEnc; ++s)
            {
                kv[i * P + j] += k[s * P + i] * v[s * P + j];
            }
        }
    }
    for (std::size_t t = 0; t < SDec; ++t)
    {
        float q[P];
        for (std::size_t p = 0; p < P; ++p)
        {
            float aq = b_q[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                aq += w_q[e * P + p] * dec[t * E + e];
            }
            q[p] = (aq < 0.0f) ? 0.0f : aq;
        }
        for (std::size_t j = 0; j < P; ++j)
        {
            float a = 0.0f;
            for (std::size_t i = 0; i < P; ++i)
            {
                a += q[i] * kv[i * P + j];
            }
            y[t * P + j] = a;
        }
    }
}

// Softmax cross-attention: Q from decoder, K/V from encoder, no mask.
template<std::size_t SDec, std::size_t SEnc, std::size_t E, std::size_t P>
void floatCrossSoftmaxReference(const float* dec, const float* mem,
                                const float* w, const float* b, float* y)
{
    const float* w_q = w;
    const float* w_k = w + E * P;
    const float* w_v = w + 2 * E * P;
    const float* b_q = b;
    const float* b_k = b + P;
    const float* b_v = b + 2 * P;

    float k[SEnc * P], v[SEnc * P];
    for (std::size_t s = 0; s < SEnc; ++s)
    {
        for (std::size_t p = 0; p < P; ++p)
        {
            float ak = b_k[p], av = b_v[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                ak += w_k[e * P + p] * mem[s * E + e];
                av += w_v[e * P + p] * mem[s * E + e];
            }
            k[s * P + p] = ak;
            v[s * P + p] = av;
        }
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(P));
    for (std::size_t t = 0; t < SDec; ++t)
    {
        float q[P];
        for (std::size_t p = 0; p < P; ++p)
        {
            float aq = b_q[p];
            for (std::size_t e = 0; e < E; ++e)
            {
                aq += w_q[e * P + p] * dec[t * E + e];
            }
            q[p] = aq;
        }
        float sc[SEnc];
        float m = -1.0e30f;
        for (std::size_t j = 0; j < SEnc; ++j)
        {
            float a = 0.0f;
            for (std::size_t p = 0; p < P; ++p)
            {
                a += q[p] * k[j * P + p];
            }
            sc[j] = a * inv;
            if (sc[j] > m) m = sc[j];
        }
        float sum = 0.0f;
        for (std::size_t j = 0; j < SEnc; ++j)
        {
            sc[j] = std::exp(sc[j] - m);
            sum += sc[j];
        }
        for (std::size_t p = 0; p < P; ++p)
        {
            float a = 0.0f;
            for (std::size_t j = 0; j < SEnc; ++j)
            {
                a += (sc[j] / sum) * v[j * P + p];
            }
            y[t * P + p] = a;
        }
    }
}

} // namespace

BOOST_AUTO_TEST_CASE(qcausal_linear_parity_with_float_reference)
{
    constexpr std::size_t S = 5, E = 6, P = 4;

    float x[S * E];
    for (std::size_t i = 0; i < S * E; ++i)
        x[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 7) - 3);
    float w[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
        w[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 11) - 5);
    float b[3 * P];
    for (std::size_t i = 0; i < 3 * P; ++i)
        b[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 3) - 1);

    float y_ref[S * P];
    floatCausalLinearAttentionReference<S, E, P>(x, w, b, y_ref);

    // Re-project to observe per-tensor ranges. The cumulative KV peaks at the
    // last position (== the full-sequence sum), so the bidirectional KV range
    // is a safe calibration upper bound for every prefix.
    float q_buf[S * P], k_buf[S * P], v_buf[S * P], kv_buf[P * P];
    {
        const float* wq = w; const float* wk = w + E * P; const float* wv = w + 2 * E * P;
        const float* bq = b; const float* bk = b + P;     const float* bv = b + 2 * P;
        for (std::size_t t = 0; t < S; ++t)
            for (std::size_t p = 0; p < P; ++p)
            {
                float aq = bq[p], ak = bk[p], av = bv[p];
                for (std::size_t e = 0; e < E; ++e)
                {
                    aq += wq[e * P + p] * x[t * E + e];
                    ak += wk[e * P + p] * x[t * E + e];
                    av += wv[e * P + p] * x[t * E + e];
                }
                q_buf[t * P + p] = (aq < 0.0f) ? 0.0f : aq;
                k_buf[t * P + p] = (ak < 0.0f) ? 0.0f : ak;
                v_buf[t * P + p] = av;
            }
        for (std::size_t i = 0; i < P; ++i)
            for (std::size_t j = 0; j < P; ++j)
            {
                float a = 0.0f;
                for (std::size_t t = 0; t < S; ++t) a += k_buf[t * P + i] * v_buf[t * P + j];
                kv_buf[i * P + j] = a;
            }
    }

    const float x_scale  = absmaxBuf(x, S * E) / 127.0f;
    const float wq_scale = absmaxBuf(w,             E * P) / 127.0f;
    const float wk_scale = absmaxBuf(w + E * P,     E * P) / 127.0f;
    const float wv_scale = absmaxBuf(w + 2 * E * P, E * P) / 127.0f;
    const float q_scale  = absmaxBuf(q_buf,  S * P) / 127.0f;
    const float k_scale  = absmaxBuf(k_buf,  S * P) / 127.0f;
    const float v_scale  = absmaxBuf(v_buf,  S * P) / 127.0f;
    const float kv_scale = absmaxBuf(kv_buf, P * P) / 127.0f;
    const float y_scale  = absmaxBuf(y_ref,  S * P) / 127.0f;

    int8_t x_q[S * E];
    int8_t w_q[3 * E * P];
    int32_t b_q[3 * P];
    quantizeBuffer<int8_t>(x, x_q, S * E, x_scale, 0, -128, 127);
    quantizeSymToI8(w,             E * P, wq_scale, w_q);
    quantizeSymToI8(w + E * P,     E * P, wk_scale, w_q + E * P);
    quantizeSymToI8(w + 2 * E * P, E * P, wv_scale, w_q + 2 * E * P);
    for (std::size_t p = 0; p < P; ++p)
    {
        b_q[p] = static_cast<int32_t>(std::lround(static_cast<double>(b[p]) /
            (static_cast<double>(x_scale) * static_cast<double>(wq_scale))));
        b_q[P + p] = static_cast<int32_t>(std::lround(static_cast<double>(b[P + p]) /
            (static_cast<double>(x_scale) * static_cast<double>(wk_scale))));
        b_q[2 * P + p] = static_cast<int32_t>(std::lround(static_cast<double>(b[2 * P + p]) /
            (static_cast<double>(x_scale) * static_cast<double>(wv_scale))));
    }

    tinymind::QCausalAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
                                 S, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.input_zero_point = 0;
    attn.q_zero_point = 0; attn.k_zero_point = 0; attn.v_zero_point = 0;
    attn.kv_zero_point = 0;
    attn.q_requantizer = buildRequantizer<int8_t>(x_scale, wq_scale, q_scale, 0, 0, 127);
    attn.k_requantizer = buildRequantizer<int8_t>(x_scale, wk_scale, k_scale, 0, 0, 127);
    attn.v_requantizer = buildRequantizer<int8_t>(x_scale, wv_scale, v_scale, 0, -128, 127);
    attn.kv_requantizer = buildRequantizer<int8_t>(k_scale, v_scale, kv_scale, 0, -128, 127);
    attn.output_requantizer = buildRequantizer<int8_t>(q_scale, kv_scale, y_scale, 0, -128, 127);

    decltype(attn)::KVState state;
    int8_t q_sc[S * P], k_sc[S * P], v_sc[S * P], kv_sc[P * P], y_q[S * P];
    attn.forward(x_q, state, q_sc, k_sc, v_sc, kv_sc, y_q);

    const float y_ref_max = absmaxBuf(y_ref, S * P);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < S * P; ++i)
    {
        const float y_back = dequantize<int8_t>(y_q[i], y_scale, 0);
        const float e = std::fabs(y_back - y_ref[i]);
        if (e > max_err) max_err = e;
    }
    BOOST_TEST(max_err < 0.10f * y_ref_max);
}

BOOST_AUTO_TEST_CASE(qcausal_linear_step_equals_forward)
{
    // Incremental decode must reproduce the full-sequence pass byte-for-byte.
    constexpr std::size_t S = 6, E = 4, P = 3;

    int8_t x_q[S * E];
    for (std::size_t i = 0; i < S * E; ++i)
        x_q[i] = static_cast<int8_t>((static_cast<int>(i) * 7 % 31) - 15);
    int8_t w_q[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
        w_q[i] = static_cast<int8_t>((static_cast<int>(i) * 5 % 17) - 8);
    int32_t b_q[3 * P] = {0};

    tinymind::QCausalAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
                                 S, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.input_zero_point = 0;
    attn.q_zero_point = 0; attn.k_zero_point = 0; attn.v_zero_point = 0;
    attn.kv_zero_point = 0;
    attn.q_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.k_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.v_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.kv_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.output_requantizer = makeI8Requant(kQ31One, 6, 0);

    decltype(attn)::KVState fwd_state;
    int8_t q_sc[S * P], k_sc[S * P], v_sc[S * P], kv_sc[P * P], y_fwd[S * P];
    attn.forward(x_q, fwd_state, q_sc, k_sc, v_sc, kv_sc, y_fwd);

    decltype(attn)::KVState step_state;
    step_state.reset();
    int8_t y_step[S * P];
    int8_t q_r[P], k_r[P], v_r[P], kv_r[P * P];
    for (std::size_t t = 0; t < S; ++t)
    {
        attn.step(x_q + t * E, step_state, q_r, k_r, v_r, kv_r, y_step + t * P);
    }

    for (std::size_t i = 0; i < S * P; ++i)
    {
        BOOST_TEST(static_cast<int>(y_step[i]) == static_cast<int>(y_fwd[i]));
    }
}

BOOST_AUTO_TEST_CASE(qcausal_softmax_parity_with_float_reference)
{
    constexpr std::size_t S = 5, E = 6, P = 4;

    float x[S * E];
    for (std::size_t i = 0; i < S * E; ++i)
        x[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 5) - 2);
    float w[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
        w[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 9) - 4);
    float b[3 * P] = {0};

    float y_ref[S * P];
    floatCausalSoftmaxAttentionReference<S, E, P>(x, w, b, y_ref);

    float q_buf[S * P], k_buf[S * P], v_buf[S * P], scores_buf[S * S];
    {
        const float* wq = w; const float* wk = w + E * P; const float* wv = w + 2 * E * P;
        for (std::size_t t = 0; t < S; ++t)
            for (std::size_t p = 0; p < P; ++p)
            {
                float aq = 0.0f, ak = 0.0f, av = 0.0f;
                for (std::size_t e = 0; e < E; ++e)
                {
                    aq += wq[e * P + p] * x[t * E + e];
                    ak += wk[e * P + p] * x[t * E + e];
                    av += wv[e * P + p] * x[t * E + e];
                }
                q_buf[t * P + p] = aq; k_buf[t * P + p] = ak; v_buf[t * P + p] = av;
            }
        const float inv = 1.0f / std::sqrt(static_cast<float>(P));
        for (std::size_t i = 0; i < S; ++i)
            for (std::size_t j = 0; j < S; ++j)
            {
                float a = 0.0f;
                for (std::size_t p = 0; p < P; ++p) a += q_buf[i * P + p] * k_buf[j * P + p];
                scores_buf[i * S + j] = a * inv;
            }
    }

    const float x_scale  = absmaxBuf(x, S * E) / 127.0f;
    const float wq_scale = absmaxBuf(w,             E * P) / 127.0f;
    const float wk_scale = absmaxBuf(w + E * P,     E * P) / 127.0f;
    const float wv_scale = absmaxBuf(w + 2 * E * P, E * P) / 127.0f;
    const float q_scale  = absmaxBuf(q_buf, S * P) / 127.0f;
    const float k_scale  = absmaxBuf(k_buf, S * P) / 127.0f;
    const float v_scale  = absmaxBuf(v_buf, S * P) / 127.0f;
    const float score_scale = absmaxBuf(scores_buf, S * S) / 127.0f;
    const float attn_scale  = 1.0f / 256.0f;
    const float y_scale  = absmaxBuf(y_ref, S * P) / 127.0f;

    int8_t x_q[S * E];
    int8_t w_q[3 * E * P];
    int32_t b_q[3 * P] = {0};
    quantizeBuffer<int8_t>(x, x_q, S * E, x_scale, 0, -128, 127);
    quantizeSymToI8(w,             E * P, wq_scale, w_q);
    quantizeSymToI8(w + E * P,     E * P, wk_scale, w_q + E * P);
    quantizeSymToI8(w + 2 * E * P, E * P, wv_scale, w_q + 2 * E * P);

    int32_t exp_lut[256];
    buildQSoftmaxExpLUT(score_scale, exp_lut);

    tinymind::QCausalAttentionSoftmax1D<int8_t, int8_t, int32_t,
                                        int8_t, int8_t, int8_t, int8_t,
                                        S, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.input_zero_point = 0;
    attn.q_zero_point = 0; attn.k_zero_point = 0; attn.v_zero_point = 0;
    attn.attn_zero_point = -128;
    attn.q_requantizer = buildRequantizer<int8_t>(x_scale, wq_scale, q_scale, 0, -128, 127);
    attn.k_requantizer = buildRequantizer<int8_t>(x_scale, wk_scale, k_scale, 0, -128, 127);
    attn.v_requantizer = buildRequantizer<int8_t>(x_scale, wv_scale, v_scale, 0, -128, 127);
    {
        Requantizer<int32_t, int8_t> r;
        const double ratio = (static_cast<double>(q_scale) * static_cast<double>(k_scale)) /
            static_cast<double>(score_scale) * qAttentionInvSqrt(P);
        int32_t mult = 0, shft = 0;
        quantizeMultiplier(ratio, mult, shft);
        r.multiplier = mult; r.shift = shft; r.zero_point = 0; r.qmin = -128; r.qmax = 127;
        attn.score_requantizer = r;
    }
    attn.softmax_exp_lut = exp_lut;
    attn.attn_qmin = -128; attn.attn_qmax = 127;
    attn.output_requantizer = buildRequantizer<int8_t>(attn_scale, v_scale, y_scale, 0, -128, 127);

    decltype(attn)::KVCache cache;
    int8_t q_sc[S * P], k_sc[S * P], v_sc[S * P];
    int8_t score_sc[S], attn_sc[S], y_q[S * P];
    attn.forward(x_q, cache, q_sc, k_sc, v_sc, score_sc, attn_sc, y_q);

    const float y_ref_max = absmaxBuf(y_ref, S * P);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < S * P; ++i)
    {
        const float y_back = dequantize<int8_t>(y_q[i], y_scale, 0);
        const float e = std::fabs(y_back - y_ref[i]);
        if (e > max_err) max_err = e;
    }
    BOOST_TEST(max_err < 0.16f * y_ref_max);
}

BOOST_AUTO_TEST_CASE(qcausal_softmax_step_equals_forward)
{
    constexpr std::size_t S = 6, E = 4, P = 3;

    int8_t x_q[S * E];
    for (std::size_t i = 0; i < S * E; ++i)
        x_q[i] = static_cast<int8_t>((static_cast<int>(i) * 7 % 31) - 15);
    int8_t w_q[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
        w_q[i] = static_cast<int8_t>((static_cast<int>(i) * 5 % 17) - 8);
    int32_t b_q[3 * P] = {0};
    int32_t exp_lut[256];
    buildQSoftmaxExpLUT(0.5f, exp_lut);

    tinymind::QCausalAttentionSoftmax1D<int8_t, int8_t, int32_t,
                                        int8_t, int8_t, int8_t, int8_t,
                                        S, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.input_zero_point = 0;
    attn.q_zero_point = 0; attn.k_zero_point = 0; attn.v_zero_point = 0;
    attn.attn_zero_point = -128;
    attn.q_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.k_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.v_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.score_requantizer = makeI8Requant(kQ31One, 6, 0);
    attn.softmax_exp_lut = exp_lut;
    attn.attn_qmin = -128; attn.attn_qmax = 127;
    attn.output_requantizer = makeI8Requant(kQ31One, 6, 0);

    decltype(attn)::KVCache fwd_cache;
    int8_t q_sc[S * P], k_sc[S * P], v_sc[S * P];
    int8_t score_sc[S], attn_sc[S], y_fwd[S * P];
    attn.forward(x_q, fwd_cache, q_sc, k_sc, v_sc, score_sc, attn_sc, y_fwd);

    decltype(attn)::KVCache step_cache;
    step_cache.reset();
    int8_t y_step[S * P];
    int8_t q_r[P], k_r[P], v_r[P], sc_r[S], at_r[S];
    for (std::size_t t = 0; t < S; ++t)
    {
        attn.step(x_q + t * E, step_cache, q_r, k_r, v_r, sc_r, at_r, y_step + t * P);
    }

    for (std::size_t i = 0; i < S * P; ++i)
    {
        BOOST_TEST(static_cast<int>(y_step[i]) == static_cast<int>(y_fwd[i]));
    }
}

BOOST_AUTO_TEST_CASE(qcross_linear_parity_with_float_reference)
{
    constexpr std::size_t SDec = 4, SEnc = 5, E = 6, P = 4;

    float dec[SDec * E], mem[SEnc * E];
    for (std::size_t i = 0; i < SDec * E; ++i)
        dec[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 7) - 3);
    for (std::size_t i = 0; i < SEnc * E; ++i)
        mem[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 5) - 2);
    float w[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
        w[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 11) - 5);
    float b[3 * P];
    for (std::size_t i = 0; i < 3 * P; ++i)
        b[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 3) - 1);

    float y_ref[SDec * P];
    floatCrossLinearReference<SDec, SEnc, E, P>(dec, mem, w, b, y_ref);

    float q_buf[SDec * P], k_buf[SEnc * P], v_buf[SEnc * P], kv_buf[P * P];
    {
        const float* wq = w; const float* wk = w + E * P; const float* wv = w + 2 * E * P;
        const float* bq = b; const float* bk = b + P;     const float* bv = b + 2 * P;
        for (std::size_t s = 0; s < SEnc; ++s)
            for (std::size_t p = 0; p < P; ++p)
            {
                float ak = bk[p], av = bv[p];
                for (std::size_t e = 0; e < E; ++e)
                {
                    ak += wk[e * P + p] * mem[s * E + e];
                    av += wv[e * P + p] * mem[s * E + e];
                }
                k_buf[s * P + p] = (ak < 0.0f) ? 0.0f : ak;
                v_buf[s * P + p] = av;
            }
        for (std::size_t t = 0; t < SDec; ++t)
            for (std::size_t p = 0; p < P; ++p)
            {
                float aq = bq[p];
                for (std::size_t e = 0; e < E; ++e) aq += wq[e * P + p] * dec[t * E + e];
                q_buf[t * P + p] = (aq < 0.0f) ? 0.0f : aq;
            }
        for (std::size_t i = 0; i < P; ++i)
            for (std::size_t j = 0; j < P; ++j)
            {
                float a = 0.0f;
                for (std::size_t s = 0; s < SEnc; ++s) a += k_buf[s * P + i] * v_buf[s * P + j];
                kv_buf[i * P + j] = a;
            }
    }

    const float d_scale  = absmaxBuf(dec, SDec * E) / 127.0f;
    const float m_scale  = absmaxBuf(mem, SEnc * E) / 127.0f;
    // Single input scale shared by Q (decoder) and K/V (encoder) projections.
    const float in_scale = (d_scale > m_scale) ? d_scale : m_scale;
    const float wq_scale = absmaxBuf(w,             E * P) / 127.0f;
    const float wk_scale = absmaxBuf(w + E * P,     E * P) / 127.0f;
    const float wv_scale = absmaxBuf(w + 2 * E * P, E * P) / 127.0f;
    const float q_scale  = absmaxBuf(q_buf,  SDec * P) / 127.0f;
    const float k_scale  = absmaxBuf(k_buf,  SEnc * P) / 127.0f;
    const float v_scale  = absmaxBuf(v_buf,  SEnc * P) / 127.0f;
    const float kv_scale = absmaxBuf(kv_buf, P * P) / 127.0f;
    const float y_scale  = absmaxBuf(y_ref,  SDec * P) / 127.0f;

    int8_t dec_q[SDec * E], mem_q[SEnc * E];
    int8_t w_q[3 * E * P];
    int32_t b_q[3 * P];
    quantizeBuffer<int8_t>(dec, dec_q, SDec * E, in_scale, 0, -128, 127);
    quantizeBuffer<int8_t>(mem, mem_q, SEnc * E, in_scale, 0, -128, 127);
    quantizeSymToI8(w,             E * P, wq_scale, w_q);
    quantizeSymToI8(w + E * P,     E * P, wk_scale, w_q + E * P);
    quantizeSymToI8(w + 2 * E * P, E * P, wv_scale, w_q + 2 * E * P);
    for (std::size_t p = 0; p < P; ++p)
    {
        b_q[p] = static_cast<int32_t>(std::lround(static_cast<double>(b[p]) /
            (static_cast<double>(in_scale) * static_cast<double>(wq_scale))));
        b_q[P + p] = static_cast<int32_t>(std::lround(static_cast<double>(b[P + p]) /
            (static_cast<double>(in_scale) * static_cast<double>(wk_scale))));
        b_q[2 * P + p] = static_cast<int32_t>(std::lround(static_cast<double>(b[2 * P + p]) /
            (static_cast<double>(in_scale) * static_cast<double>(wv_scale))));
    }

    tinymind::QCrossAttention1D<int8_t, int8_t, int32_t, int8_t, int8_t, int8_t,
                                SDec, SEnc, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.q_input_zero_point = 0; attn.kv_input_zero_point = 0;
    attn.q_zero_point = 0; attn.k_zero_point = 0; attn.v_zero_point = 0;
    attn.kv_zero_point = 0;
    attn.q_requantizer = buildRequantizer<int8_t>(in_scale, wq_scale, q_scale, 0, 0, 127);
    attn.k_requantizer = buildRequantizer<int8_t>(in_scale, wk_scale, k_scale, 0, 0, 127);
    attn.v_requantizer = buildRequantizer<int8_t>(in_scale, wv_scale, v_scale, 0, -128, 127);
    attn.kv_requantizer = buildRequantizer<int8_t>(k_scale, v_scale, kv_scale, 0, -128, 127);
    attn.output_requantizer = buildRequantizer<int8_t>(q_scale, kv_scale, y_scale, 0, -128, 127);

    int8_t k_sc[SEnc * P], v_sc[SEnc * P], kv_sc[P * P], q_sc[SDec * P], y_q[SDec * P];
    attn.forward(dec_q, mem_q, k_sc, v_sc, kv_sc, q_sc, y_q);

    const float y_ref_max = absmaxBuf(y_ref, SDec * P);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < SDec * P; ++i)
    {
        const float y_back = dequantize<int8_t>(y_q[i], y_scale, 0);
        const float e = std::fabs(y_back - y_ref[i]);
        if (e > max_err) max_err = e;
    }
    BOOST_TEST(max_err < 0.12f * y_ref_max);
}

BOOST_AUTO_TEST_CASE(qcross_softmax_parity_with_float_reference)
{
    constexpr std::size_t SDec = 4, SEnc = 5, E = 6, P = 4;

    float dec[SDec * E], mem[SEnc * E];
    for (std::size_t i = 0; i < SDec * E; ++i)
        dec[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 7) - 3);
    for (std::size_t i = 0; i < SEnc * E; ++i)
        mem[i] = 0.1f * static_cast<float>((static_cast<int>(i) % 5) - 2);
    float w[3 * E * P];
    for (std::size_t i = 0; i < 3 * E * P; ++i)
        w[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 9) - 4);
    // Non-zero biases keep the softmax-weighted value average off zero; with
    // these symmetric inputs a bias-free projection would cancel to ~0 and the
    // relative-error tolerance would be meaningless.
    float b[3 * P];
    for (std::size_t i = 0; i < 3 * P; ++i)
        b[i] = 0.05f * static_cast<float>((static_cast<int>(i) % 3) + 1);

    float y_ref[SDec * P];
    floatCrossSoftmaxReference<SDec, SEnc, E, P>(dec, mem, w, b, y_ref);

    float q_buf[SDec * P], k_buf[SEnc * P], v_buf[SEnc * P], scores_buf[SDec * SEnc];
    {
        const float* wq = w; const float* wk = w + E * P; const float* wv = w + 2 * E * P;
        const float* bq = b; const float* bk = b + P;     const float* bv = b + 2 * P;
        for (std::size_t s = 0; s < SEnc; ++s)
            for (std::size_t p = 0; p < P; ++p)
            {
                float ak = bk[p], av = bv[p];
                for (std::size_t e = 0; e < E; ++e)
                {
                    ak += wk[e * P + p] * mem[s * E + e];
                    av += wv[e * P + p] * mem[s * E + e];
                }
                k_buf[s * P + p] = ak; v_buf[s * P + p] = av;
            }
        for (std::size_t t = 0; t < SDec; ++t)
            for (std::size_t p = 0; p < P; ++p)
            {
                float aq = bq[p];
                for (std::size_t e = 0; e < E; ++e) aq += wq[e * P + p] * dec[t * E + e];
                q_buf[t * P + p] = aq;
            }
        const float inv = 1.0f / std::sqrt(static_cast<float>(P));
        for (std::size_t t = 0; t < SDec; ++t)
            for (std::size_t j = 0; j < SEnc; ++j)
            {
                float a = 0.0f;
                for (std::size_t p = 0; p < P; ++p) a += q_buf[t * P + p] * k_buf[j * P + p];
                scores_buf[t * SEnc + j] = a * inv;
            }
    }

    const float d_scale  = absmaxBuf(dec, SDec * E) / 127.0f;
    const float m_scale  = absmaxBuf(mem, SEnc * E) / 127.0f;
    const float in_scale = (d_scale > m_scale) ? d_scale : m_scale;
    const float wq_scale = absmaxBuf(w,             E * P) / 127.0f;
    const float wk_scale = absmaxBuf(w + E * P,     E * P) / 127.0f;
    const float wv_scale = absmaxBuf(w + 2 * E * P, E * P) / 127.0f;
    const float q_scale  = absmaxBuf(q_buf, SDec * P) / 127.0f;
    const float k_scale  = absmaxBuf(k_buf, SEnc * P) / 127.0f;
    const float v_scale  = absmaxBuf(v_buf, SEnc * P) / 127.0f;
    const float score_scale = absmaxBuf(scores_buf, SDec * SEnc) / 127.0f;
    const float attn_scale  = 1.0f / 256.0f;
    const float y_scale  = absmaxBuf(y_ref, SDec * P) / 127.0f;

    int8_t dec_q[SDec * E], mem_q[SEnc * E];
    int8_t w_q[3 * E * P];
    int32_t b_q[3 * P];
    quantizeBuffer<int8_t>(dec, dec_q, SDec * E, in_scale, 0, -128, 127);
    quantizeBuffer<int8_t>(mem, mem_q, SEnc * E, in_scale, 0, -128, 127);
    quantizeSymToI8(w,             E * P, wq_scale, w_q);
    quantizeSymToI8(w + E * P,     E * P, wk_scale, w_q + E * P);
    quantizeSymToI8(w + 2 * E * P, E * P, wv_scale, w_q + 2 * E * P);
    for (std::size_t p = 0; p < P; ++p)
    {
        b_q[p] = static_cast<int32_t>(std::lround(static_cast<double>(b[p]) /
            (static_cast<double>(in_scale) * static_cast<double>(wq_scale))));
        b_q[P + p] = static_cast<int32_t>(std::lround(static_cast<double>(b[P + p]) /
            (static_cast<double>(in_scale) * static_cast<double>(wk_scale))));
        b_q[2 * P + p] = static_cast<int32_t>(std::lround(static_cast<double>(b[2 * P + p]) /
            (static_cast<double>(in_scale) * static_cast<double>(wv_scale))));
    }

    int32_t exp_lut[256];
    buildQSoftmaxExpLUT(score_scale, exp_lut);

    tinymind::QCrossAttentionSoftmax1D<int8_t, int8_t, int32_t,
                                       int8_t, int8_t, int8_t, int8_t,
                                       SDec, SEnc, E, P> attn;
    attn.weights = w_q;
    attn.biases  = b_q;
    attn.q_input_zero_point = 0; attn.kv_input_zero_point = 0;
    attn.q_zero_point = 0; attn.k_zero_point = 0; attn.v_zero_point = 0;
    attn.attn_zero_point = -128;
    attn.q_requantizer = buildRequantizer<int8_t>(in_scale, wq_scale, q_scale, 0, -128, 127);
    attn.k_requantizer = buildRequantizer<int8_t>(in_scale, wk_scale, k_scale, 0, -128, 127);
    attn.v_requantizer = buildRequantizer<int8_t>(in_scale, wv_scale, v_scale, 0, -128, 127);
    {
        Requantizer<int32_t, int8_t> r;
        const double ratio = (static_cast<double>(q_scale) * static_cast<double>(k_scale)) /
            static_cast<double>(score_scale) * qAttentionInvSqrt(P);
        int32_t mult = 0, shft = 0;
        quantizeMultiplier(ratio, mult, shft);
        r.multiplier = mult; r.shift = shft; r.zero_point = 0; r.qmin = -128; r.qmax = 127;
        attn.score_requantizer = r;
    }
    attn.softmax_exp_lut = exp_lut;
    attn.attn_qmin = -128; attn.attn_qmax = 127;
    attn.output_requantizer = buildRequantizer<int8_t>(attn_scale, v_scale, y_scale, 0, -128, 127);

    decltype(attn)::KVCache cache;
    int8_t q_sc[SDec * P], score_sc[SEnc], attn_sc[SEnc], y_q[SDec * P];
    attn.forward(dec_q, mem_q, cache, q_sc, score_sc, attn_sc, y_q);

    const float y_ref_max = absmaxBuf(y_ref, SDec * P);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < SDec * P; ++i)
    {
        const float y_back = dequantize<int8_t>(y_q[i], y_scale, 0);
        const float e = std::fabs(y_back - y_ref[i]);
        if (e > max_err) max_err = e;
    }
    BOOST_TEST(max_err < 0.16f * y_ref_max);
}

BOOST_AUTO_TEST_SUITE_END()
