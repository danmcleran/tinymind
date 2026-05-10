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
#include "qconv2d.hpp"
#include "qpool2d.hpp"
#include "qdepthwiseconv2d.hpp"
#include "qpointwiseconv2d.hpp"
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
using tinymind::QConv2D;
using tinymind::QMaxPool2D;
using tinymind::QAvgPool2D;
using tinymind::QGlobalAvgPool2D;
using tinymind::QDepthwiseConv2D;
using tinymind::QPointwiseConv2D;
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

#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD

BOOST_AUTO_TEST_SUITE_END()
